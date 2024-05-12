import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.optim.lr_scheduler import PolynomialLR, ReduceLROnPlateau
from graphormer.cli import LossReductionType
from graphormer.schedulers import GreedyLR
from graphormer.config.utils import calculate_pos_weight, model_init_print, save_checkpoint
from graphormer.config.hparams import HyperparameterConfig


def train_model(
        hparam_config: HyperparameterConfig,
    ):
    logging_config = hparam_config.logging_config()
    data_config = hparam_config.data_config()
    model_config = hparam_config.model_config()
    loss_config = hparam_config.loss_config()
    optimizer_config = hparam_config.optimizer_config()
    scheduler_config = hparam_config.scheduler_config()

    writer = logging_config.build()
    train_loader, test_loader = data_config.build()
    assert data_config.num_node_features is not None
    assert data_config.num_edge_features is not None
    device = torch.device(hparam_config.torch_device)
    model = model_config.with_node_feature_dim(data_config.num_node_features).with_edge_feature_dim(data_config.num_edge_features).with_output_dim(1).build().to(device)
    pos_weight = calculate_pos_weight(train_loader)
    loss_function = loss_config.with_pos_weight(pos_weight).build()
    optimizer = optimizer_config.build(model)
    scheduler = hparam_config.scheduler_config().build(optimizer)
    effective_batch_size = scheduler_config.effective_batch_size
    epochs = hparam_config.epochs
    start_epoch = hparam_config.start_epoch
    accumulation_steps = optimizer_config.accumulation_steps
    loss_reduction = optimizer_config.loss_reduction_type

    model_init_print(hparam_config, model)

    progress_bar = tqdm(total=0, desc="Initializing...", unit="batch")
    train_batches_per_epoch = len(train_loader)
    eval_batches_per_epoch = len(test_loader)
    for epoch in range(start_epoch, epochs):
        total_train_loss = 0.0
        total_eval_loss = 0.0

        # Set total length for training phase and update description
        progress_bar.reset(total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Train")

        model.train()
        avg_loss = 0.0
        train_batch_num = epoch * train_batches_per_epoch
        for batch_idx, batch in enumerate(train_loader):
            batch.to(device)
            y = batch.y.to(device)

            if train_batch_num == 0:
                writer.add_graph(
                    model, [batch.x, batch.edge_index, batch.edge_attr,
                            batch.ptr, batch.node_paths, batch.edge_paths]
                )

            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.ptr,
                batch.node_paths,
                batch.edge_paths,
            )

            loss = loss_function(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparam_config.clip_grad_norm, error_if_nonfinite=True)

            # FIX: Fix scaling of the last batch
            if should_step(batch_idx, accumulation_steps, train_batches_per_epoch):
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            writer.add_scalar("train/batch_loss", batch_loss, train_batch_num)
            writer.add_scalar("train/sample_loss", batch_loss /
                              output.shape[0] if loss_reduction == LossReductionType.SUM else batch_loss, train_batch_num)
            total_train_loss += batch_loss

            avg_loss = total_train_loss / (progress_bar.n + 1)
            writer.add_scalar("train/avg_loss", avg_loss, train_batch_num)

            progress_bar.set_postfix_str(f"Avg Loss: {avg_loss:.4f}")
            progress_bar.update()  # Increment the progress bar
            train_batch_num += 1
        if isinstance(scheduler, PolynomialLR):
            scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0] * accumulation_steps if loss_reduction == LossReductionType.MEAN else scheduler.get_last_lr()[0] * effective_batch_size, epoch)

        # Prepare for the evaluation phase
        progress_bar.reset(total=len(test_loader))
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Eval")

        all_eval_labels = []
        all_eval_preds = []

        model.eval()
        eval_batch_num = epoch * eval_batches_per_epoch
        for batch in test_loader:
            batch.to(device)
            y = batch.y.to(device)
            with torch.no_grad():
                output = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.ptr,
                    batch.node_paths,
                    batch.edge_paths,
                )
                loss = loss_function(output, y)
            batch_loss: float = loss.item()
            writer.add_scalar("eval/batch_loss", batch_loss, eval_batch_num)
            total_eval_loss += batch_loss

            eval_preds = torch.round(torch.sigmoid(output)).tolist()
            eval_labels = y.cpu().numpy()
            if sum(eval_labels) > 0:
                batch_bac = balanced_accuracy_score(eval_labels, eval_preds)
                writer.add_scalar("eval/batch_bac", batch_bac, eval_batch_num)

            all_eval_preds.extend(eval_preds)
            all_eval_labels.extend(eval_labels)

            progress_bar.update()  # Manually increment for each batch in eval
            eval_batch_num += 1

        if isinstance(scheduler, (ReduceLROnPlateau, GreedyLR)):
            scheduler.step(total_eval_loss)

        avg_eval_loss = total_eval_loss / len(test_loader)
        progress_bar.set_postfix_str(f"Avg Eval Loss: {avg_eval_loss:.4f}")
        bac = balanced_accuracy_score(all_eval_labels, all_eval_preds)
        ac = accuracy_score(all_eval_labels, all_eval_preds)
        bac_adj = balanced_accuracy_score(
            all_eval_labels, all_eval_preds, adjusted=True)
        writer.add_scalar("eval/acc", ac, epoch)
        writer.add_scalar("eval/bac", bac, epoch)
        writer.add_scalar("eval/bac_adj", bac_adj, epoch)
        writer.add_scalar("eval/avg_eval_loss", avg_eval_loss, epoch)

        print(
            f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f} | Avg Eval Loss: {
                avg_eval_loss:.4f} | Eval BAC: {bac:.4f} | Eval ACC: {ac:.4f}"
        )

        if epoch % hparam_config.checkpt_save_interval == 0:
            save_checkpoint(
                epoch,
                hparam_config,
                model,
                optimizer,
                loss_function,
                scheduler,
            )

    progress_bar.close()


def should_step(batch_idx: int, accumulation_steps: int, train_batches_per_epoch: int) -> bool:
    if accumulation_steps <= 1:
        return True
    if batch_idx > 0 and (batch_idx + 1) % accumulation_steps == 0:
        return True
    if batch_idx >= train_batches_per_epoch - 1:
        return True
    return False
