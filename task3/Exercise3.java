package mlrwd.task3;

import uk.ac.cam.cl.mlrwd.utils.*;
import uk.ac.cam.cl.mlrwd.utils.BestFit.*;
import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.util.*;
import java.util.stream.*;
import java.nio.file.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;

public class Exercise3 {
	int xMax;
	private List<String> tokens = new LinkedList<>();
	private List<FrequencyRanking> wordList;
	private List<Point> logPoints;
	private Line bestFit;

	public Exercise3(Path dir, int max) throws IOException {
		xMax = max;
		loadDataSet(dir);
		processDataSet();
		processLogs();
	}

	public double predict(double rank) {
		return bestFit.yIntercept + Math.log(rank) * bestFit.gradient;
	}
	
	public void plotLinear() {
		ChartPlotter.plotLines(
			wordList.stream()
				.map(f -> new Point(f.ranking, f.frequency))
				.collect(Collectors.toList())
		);
	}

	public void plotLinear(Set<String> keyWords) {
		ChartPlotter.plotLines(
			wordList.stream()
				.map(f -> new Point(f.ranking, f.frequency))
				.collect(Collectors.toList()),
			
			wordList.stream()
				.filter(f -> keyWords.contains(f.word))
				.map(f -> new Point(f.ranking, f.frequency))
				.collect(Collectors.toList())
		);
	}

	public void plotLog() {
		List<Point> lobf = new ArrayList<>();
		lobf.add(new Point(0, bestFit.yIntercept));
		lobf.add(new Point(
			Math.log(xMax),
			predict(xMax)
		));

		ChartPlotter.plotLines(
			logPoints, lobf
		);
	}

	public void plotHeapsLaw() {
		List<Point> points = new ArrayList<>();
		Set<String> typeSet = new HashSet<>();
		int count = 0;

		for (String token : tokens) {
			typeSet.add(token);
			++count;

			// If PoT, plot
			for (double i = 0, p = 1; p <= count; ++i, p = Math.pow(2, i)) {
				if (p == count) {
					points.add(new Point(Math.log(count), Math.log(typeSet.size())));
					break;
				}
			}
		}

		ChartPlotter.plotLines(points);
	}

	public void predictWords(Set<String> keyWords) {
		wordList.stream()
			.filter(f -> keyWords.contains(f.word))
			.forEach(f -> System.out.println("| " + f.word + "\t\t| " + Math.exp(predict(f.ranking)) + "\t\t| " + f.frequency + "\t\t| " + Math.abs(Math.exp(predict(f.ranking)) - f.frequency) / f.frequency + "\t\t|"));
	}

	public void calculateConstants() {
		System.out.println("k = " + Math.exp(bestFit.yIntercept));
		System.out.println("a = " + -bestFit.gradient);
	}
	
	private void loadDataSet(Path dir) throws IOException {
		try (DirectoryStream<Path> files = Files.newDirectoryStream(dir)) {
			for (Path item : files) {
				tokens.addAll(Tokenizer.tokenize(item));
			}
		}
		catch (IOException e) {
			throw new IOException("Can't read the reviews.", e);
		}
	}

	private void processDataSet() {
		// Load type frequencies
		Map<String, Integer> freqMap = new HashMap<>();
		for (String token : tokens) {
			if (freqMap.containsKey(token)) {
				freqMap.put(token, freqMap.get(token) + 1);
			}
			else {
				freqMap.put(token, 1);
			}
		}

		// Create an ordered list of FrequencyRanking (word, frequency, rank) objects
		AtomicInteger i = new AtomicInteger(1);
		this.wordList = freqMap.entrySet().stream()
			.sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
			.map(entry -> new FrequencyRanking(entry.getKey(), entry.getValue(), i.getAndIncrement()))
			.limit(xMax)
			.collect(Collectors.toList());
	}

	private void processLogs() {
		this.logPoints = wordList.stream()
			.map(f -> new Point(Math.log(f.ranking), Math.log(f.frequency)))
			.collect(Collectors.toList());

		this.bestFit = BestFit.leastSquares(
			this.logPoints.stream()
				.collect(Collectors.toMap(p -> p, p -> Math.log(p.y)))
		);
	}

	public static class FrequencyRanking {
		public String word;
		public int frequency, ranking;
		public FrequencyRanking(String w, int f, int r) {
			word = w;
			frequency = f;
			ranking = r;
		}
	}

    public static void main(String[] args) throws IOException {
		Path dataset = Paths.get("data/large_dataset");
		Exercise3 ex3 = new Exercise3(dataset, 10000);

		Set<String> keyWords = new HashSet<>();
		keyWords.add("too");
		keyWords.add("relax");
		keyWords.add("don't");
		keyWords.add("better");
		keyWords.add("great");
		keyWords.add("well");
		keyWords.add("satisfying");
		keyWords.add("annoying");
		keyWords.add("recommend");
		keyWords.add("interesting");

		ex3.plotLinear(keyWords);
		ex3.plotLog();
		ex3.plotHeapsLaw();

		ex3.predictWords(keyWords);
		ex3.calculateConstants();
    }
}