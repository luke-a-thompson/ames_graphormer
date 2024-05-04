use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::{Bfs, IntoNodeIdentifiers};
use pyo3::prelude::*;
use std::collections::hash_set::HashSet;

#[derive(Debug, Clone)]
struct Paths<const N: usize> {
    node_paths: Vec<[i16; N]>,
    edge_paths: Vec<[i16; N]>,
}

fn shortest_paths_from_source<const N: usize>(
    g: &UnGraph<u32, ()>,
    source: NodeIndex,
    paths: &mut Paths<N>,
) {
    let num_nodes = g.node_count();
    if num_nodes == 0 {
        return;
    }
    paths.node_paths[source.index()][0] = source.index() as i16;
    let mut bfs = Bfs::new(g, source);
    let mut visited = HashSet::new();
    visited.insert(source);

    while let Some(v) = bfs.next(&g) {
        for w in g.neighbors(v) {
            if visited.contains(&w) {
                continue;
            }
            visited.insert(w);
            let w_idx = w.index();
            let v_idx = v.index();
            paths.node_paths[w_idx] = paths.node_paths[v_idx];
            paths.edge_paths[w_idx] = paths.edge_paths[v_idx];
            if let Some(w_node_col_idx) = paths.node_paths[w_idx].iter().position(|&x| x == -1) {
                paths.node_paths[w_idx][w_node_col_idx] = w_idx as i16;
                if let Some(w_edge_col_idx) = paths.edge_paths[w_idx].iter().position(|&x| x == -1)
                {
                    let edge_nodes = &paths.node_paths[w_idx][w_node_col_idx - 1..=w_node_col_idx];
                    if let Some(ei) =
                        g.find_edge((edge_nodes[0] as u32).into(), (edge_nodes[1] as u32).into())
                    {
                        paths.edge_paths[w.index()][w_edge_col_idx] = ei.index() as i16;
                    }
                }
            }
        }
    }
}

#[pyfunction]
fn shortest_paths(
    edges: Vec<(u32, u32)>,
    max_path_len: usize,
) -> (Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i16>>>) {
    let g = UnGraph::<u32, ()>::from_edges(edges);

    let num_nodes = g.node_count();

    let mut paths = vec![
        Paths {
            node_paths: vec![[-1; 64]; num_nodes],
            edge_paths: vec![[-1; 64]; num_nodes],
        };
        num_nodes
    ];

    let mut node_paths = Vec::with_capacity(num_nodes);
    let mut edge_paths = Vec::with_capacity(num_nodes);

    for source in g.node_identifiers() {
        shortest_paths_from_source(&g, source, &mut paths[source.index()]);
    }

    for path in paths {
        node_paths.push(
            path.node_paths
                .into_iter()
                .map(|x| x[..max_path_len].to_vec())
                .collect(),
        );
        edge_paths.push(
            path.edge_paths
                .into_iter()
                .map(|x| x[..max_path_len].to_vec())
                .collect(),
        );
    }

    (node_paths, edge_paths)
}

#[pymodule]
fn gnn_tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shortest_paths, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_finds_shortest_paths() {
        let edges = vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 1),
            (3, 1),
            (3, 4),
            (3, 18),
            (4, 3),
            (4, 5),
            (5, 4),
            (5, 6),
            (5, 16),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (7, 9),
            (8, 7),
            (9, 7),
            (9, 10),
            (9, 14),
            (10, 9),
            (10, 11),
            (11, 10),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 9),
            (14, 13),
            (14, 15),
            (15, 14),
            (16, 5),
            (16, 17),
            (17, 16),
            (17, 18),
            (18, 3),
            (18, 17),
        ];
        let graph = UnGraph::<u32, ()>::from_edges(edges);
        let num_nodes = graph.node_count();
        let mut paths = Paths {
            node_paths: vec![[-1; 6]; num_nodes],
            edge_paths: vec![[-1; 6]; num_nodes],
        };
        shortest_paths_from_source(&graph, 0.into(), &mut paths);

        let expected_node_paths = [
            [0, -1, -1, -1, -1],
            [0, 1, -1, -1, -1],
            [0, 1, 2, -1, -1],
            [0, 1, 3, -1, -1],
            [0, 1, 3, 4, -1],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 18, 17],
            [0, 1, 3, 18, 17],
            [0, 1, 3, 18, -1],
        ];

        let shortened_node_paths: Vec<[i16; 5]> = paths
            .node_paths
            .into_iter()
            .map(|x| {
                let mut items = [0; 5];
                items.copy_from_slice(&x[..5]);
                items
            })
            .collect();

        assert_eq!(shortened_node_paths.len(), expected_node_paths.len());

        for (expected_path, shortened_path) in
            expected_node_paths.iter().zip(shortened_node_paths.iter())
        {
            assert!(expected_path
                .iter()
                .zip(shortened_path.iter())
                .all(|(a, b)| a == b));
        }

        let expected_edge_paths = [
            [-1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
            [0, 2, -1, -1, -1],
            [0, 3, -1, -1, -1],
            [0, 3, 6, -1, -1],
            [0, 3, 6, 9, -1],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 6, 9, 11],
            [0, 3, 7, 39, 36],
            [0, 3, 7, 39, -1],
            [0, 3, 7, -1, -1],
        ];

        let shortened_edge_paths: Vec<[i16; 5]> = paths
            .edge_paths
            .into_iter()
            .map(|x| {
                let mut items = [0; 5];
                items.copy_from_slice(&x[..5]);
                items
            })
            .collect();

        assert_eq!(shortened_edge_paths.len(), expected_edge_paths.len());

        for (expected_path, shortened_path) in
            expected_edge_paths.iter().zip(shortened_edge_paths.iter())
        {
            println!("{:?}", expected_path);
            println!("{:?}", shortened_path);
            println!();
            assert!(expected_path
                .iter()
                .zip(shortened_path.iter())
                .all(|(a, b)| a == b));
        }
    }
}
