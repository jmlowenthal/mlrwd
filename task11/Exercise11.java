package mlrwd.task11;

import uk.ac.cam.cl.mlrwd.exercises.social_networks.*;

import java.nio.file.*;
import java.util.*;
import java.io.IOException;

public class Exercise11 implements IExercise11 {
	public Map<Integer, Double> getNodeBetweenness(Path graphFile) throws IOException {
		// G = (V, E)
		Map<Integer, Set<Integer>> edges = loadGraph(graphFile);
		Set<Integer> vertices = new HashSet<>();
		for (Map.Entry<Integer, Set<Integer>> entry : edges.entrySet()) {
			vertices.add(entry.getKey());
			vertices.addAll(entry.getValue());
		}

		Map<Integer, Double> cb = new HashMap<>();
		for (Integer v : vertices) {
			cb.put(v, 0.0);
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
				for (Integer w : edges.get(v)) {
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
					delta.put(v, delta.get(v) + (double)sigma.get(v) / sigma.get(w) * (1 + delta.get(w)));
				}
				if (!w.equals(s)) {
					cb.put(w, cb.get(w) + delta.get(w));
				}
			}
		}


		for (Integer v : vertices) {
			cb.put(v, cb.get(v) / 2.0);
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
}