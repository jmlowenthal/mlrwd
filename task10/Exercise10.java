package mlrwd.task10;

import uk.ac.cam.cl.mlrwd.exercises.social_networks.*;

import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import java.io.IOException;

public class Exercise10 implements IExercise10 {
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
	
	public Map<Integer, Integer> getConnectivities(Map<Integer, Set<Integer>> graph) {
		return graph.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().size()));
	}
	
	public int getDiameter(Map<Integer, Set<Integer>> graph) {
		int diameter = 0;
		for (Integer start : graph.keySet()) {
			Map<Integer, Integer> distances = new HashMap<>();
			Queue<Integer> nodeQueue = new LinkedList<Integer>();
			distances.put(start, 0);
			nodeQueue.add(start);
			
			int localDiameter = 0;
			while (!nodeQueue.isEmpty()) {
				Integer current = nodeQueue.poll();
				
				int dist = distances.get(current);
				for (Integer neighbour : graph.get(current)) {
					if (!distances.containsKey(neighbour)) {
						localDiameter = dist + 1;
						distances.put(neighbour, localDiameter);
						nodeQueue.add(neighbour);
					}
				}
			}
			
			if (localDiameter > diameter) {
				diameter = localDiameter;
			}
		}
		
		return diameter;
	}
}
