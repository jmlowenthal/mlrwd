package mlrwd.task12;

import uk.ac.cam.cl.mlrwd.exercises.social_networks.*;

import java.util.*;
import java.io.IOException;
import java.nio.file.*;

public class Exercise12 implements IExercise12 {
	public List<Set<Integer>> GirvanNewman(Map<Integer, Set<Integer>> graph, int minimumComponents) {
		List<Set<Integer>> components = getComponents(graph);
		while (components.size() < minimumComponents) {
			Map<Integer, Map<Integer, Double>> betweenness = getEdgeBetweenness(graph);
			
			// Find all edges of maximum betweenness centrality
			List<Edge> edges = new ArrayList<>();
			double max = Double.NEGATIVE_INFINITY;
			for (Map.Entry<Integer, Map<Integer, Double>> entryA : betweenness.entrySet()) {
				Integer u = entryA.getKey();
				Map<Integer, Double> neighbours = entryA.getValue();
				for (Map.Entry<Integer, Double> entryB : neighbours.entrySet()) {
					Integer v = entryB.getKey();
					Double bc = entryB.getValue();
					if (Math.abs(bc - max) < 1e-06) {
						edges.add(new Edge(u, v));
					}
					else if (bc > max) {
						edges.clear();
						edges.add(new Edge(u, v));
						max = bc;
					}
				}
			}
			
			// Cut edges
			for (Edge edge : edges) {
				graph.get(edge.u).remove(edge.v);
			}

			components = getComponents(graph);
		}
		return components;
	}

	public int getNumberOfEdges(Map<Integer, Set<Integer>> graph) {
		int count = 0;
		for (Set<Integer> neighbours : graph.values()) {
			count += neighbours.size();
		}
		return count / 2;
	}

	public List<Set<Integer>> getComponents(Map<Integer, Set<Integer>> graph) {
		List<Set<Integer>> components = new ArrayList<>();
		Set<Integer> unfoundNodes = new HashSet<>(graph.keySet());
		while (!unfoundNodes.isEmpty()) {
			Set<Integer> component = new HashSet<>();
			components.add(component);

			// Depth-first search
			Stack<Integer> stack = new Stack<>();
			stack.push(unfoundNodes.iterator().next());
			while (!stack.isEmpty()) {
				Integer current = stack.pop();
				component.add(current);
				unfoundNodes.remove(current);
				for (Integer neighbour : graph.get(current)) {
					if (unfoundNodes.contains(neighbour)) stack.push(neighbour);
				}
			}
		}
		return components;
	}

	public Map<Integer, Map<Integer, Double>> getEdgeBetweenness(Map<Integer, Set<Integer>> graph) {
		// G = (V, E)
		Set<Integer> vertices = new HashSet<>();
		for (Map.Entry<Integer, Set<Integer>> entry : graph.entrySet()) {
			vertices.add(entry.getKey());
			vertices.addAll(entry.getValue());
		}

		Map<Integer, Map<Integer, Double>> cb = new HashMap<>();
		for (Integer u : vertices) {
			Map<Integer, Double> map = new HashMap<>();
			cb.put(u, map);
			for (Integer v : vertices) {
				map.put(v, 0.0);
			}
		}

		for (Integer s : vertices) {
			Queue<Integer> queue = new LinkedList<>();
			Stack<Integer> stack = new Stack<>();

			Map<Integer, Integer> dist = new HashMap<>();
			Map<Integer, List<Integer>> pred = new HashMap<>();
			Map<Integer, Integer> sigma = new HashMap<>();
			Map<Integer, Double> delta = new HashMap<>();
			
			// Intialisation
			for (Integer w : vertices) {
				pred.put(w, new ArrayList<>());
				dist.put(w, -1);
				sigma.put(w, 0);
			}

			dist.put(s, 0);
			sigma.put(s, 1);
			queue.add(s);

			while (!queue.isEmpty()) {
				Integer v = queue.remove();
				stack.push(v);
				for (Integer w : graph.get(v)) {
					// Path discovery
					if (dist.get(w) < 0) {
						dist.put(w, dist.get(v) + 1);
						queue.add(w);
					}

					// Path counting
					if (dist.get(w) == dist.get(v) + 1) {
						sigma.put(w, sigma.get(w) + sigma.get(v));
						pred.get(w).add(v);
					}
				}
			}

			// Accumulation
			for (Integer v : vertices) {
				delta.put(v, 0.0);
			}

			while (!stack.isEmpty()) {
				Integer w = stack.pop();
				for (Integer v : pred.get(w)) {
					double c = (double)sigma.get(v) / sigma.get(w) * (1 + delta.get(w));
					cb.get(v).put(w, cb.get(v).get(w) + c);
					delta.put(v, delta.get(v) + c);
				}
			}
		}

		return cb;
	}

	public Map<Integer, Set<Integer>> loadGraph(Path graphFile) throws IOException {
		Map<Integer, Set<Integer>> graph = new HashMap<>();
		try {
			Files.lines(graphFile).map(s -> s.split(" ", 2)).forEach(arr -> {
				int u = Integer.parseInt(arr[0]);
				int v = Integer.parseInt(arr[1]);
			
				if (!graph.containsKey(u)) {
					graph.put(u, new HashSet<>());
				}
			
				if (!graph.containsKey(v)) {
					graph.put(v, new HashSet<>());
				}
			
				graph.get(u).add(v);
				graph.get(v).add(u);
			});
		}
		catch (NumberFormatException e) {
			throw new IOException(e.getMessage());
		}
		return graph;
	}

	private class Edge {
		public Integer u, v;
		public Edge(Integer a, Integer b) {
			u = a; v = b;
		}
	}
}
