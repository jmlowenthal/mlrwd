package mlrwd.task1;

import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Exercise1 implements IExercise1 {
	public Map<String, Sentiment> parseLexicon(Path lexiconFile) throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(lexiconFile)) {
			Map<String, Sentiment> lexicon = new HashMap<>();
			String line;
			while ((line = reader.readLine()) != null) {
				Scanner scanner = new Scanner(line);;
				String word = null;
				while (scanner.hasNext()) {
					String pair = scanner.next();
					String[] arr = pair.split("=");
					if (arr.length == 2) {
						switch (arr[0]) {
							case "word1": {
								word = arr[1];
								break;
							}
							case "priorpolarity": {
								switch (arr[1]) {
									case "positive":
										lexicon.put(word, Sentiment.POSITIVE);
										break;
									case "negative":
										lexicon.put(word, Sentiment.NEGATIVE);
								}
								break;
							}
						}
					}
				}
			}

			return lexicon;
		}
		catch (IOException e) {
			throw new IOException("Can't access file " + lexiconFile, e);
		}
	}

	public Map<Path, Sentiment> simpleClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
		Map<String, Sentiment> lexicon = parseLexicon(lexiconFile);
		Map<Path, Sentiment> result = new HashMap<>();

		for (Path path : testSet) {
			int rating = 0;

			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				Sentiment s = lexicon.get(token);
				if (s != null) {
					switch (s) {
						case POSITIVE:
							++rating;
							break;
						case NEGATIVE:
							--rating;
							break;
					}
				}
			}

			Sentiment predict = rating < 0 ? Sentiment.NEGATIVE : Sentiment.POSITIVE;
			result.put(path, predict);
		}
		return result;
	}

	public double calculateAccuracy(Map<Path, Sentiment> trueSentiments, Map<Path, Sentiment> predictedSentiments) {
		double correct = 0;
		for (Map.Entry<Path, Sentiment> predicted : predictedSentiments.entrySet()) {
			if (predicted.getValue() == trueSentiments.get(predicted.getKey())) {
				++correct;
			}
		}
		return correct / predictedSentiments.size();
	}

	public Map<Path, Sentiment> improvedClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
		Map<String, Sentiment> lexicon = parseLexicon(lexiconFile);
		Map<Path, Sentiment> result = new HashMap<>();

		for (Path path : testSet) {
			int positive = 0, negative = 0;
			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				Sentiment s = lexicon.get(token);
				if (s != null) {
					switch (s) {
						case POSITIVE:
							++positive;
							break;
						case NEGATIVE:
							++negative;
							break;
					}
				}
			}

			// Positive words are often used in negative reviews (sarcasm) and certainly more
			// common that negative words in positive reviews
			Sentiment predict = positive * .7 < negative ? Sentiment.NEGATIVE : Sentiment.POSITIVE;
			result.put(path, predict);
		}
		return result;
	}

	public static void main(String[] args) throws IOException {
		Exercise1 ex1 = new Exercise1();
		Set<Path> paths = new HashSet<Path>();
		paths.add(Paths.get("/home/jamie/wiki/compsci/tasks/mlrwd1 - opinion1.txt"));
		System.out.println(ex1.improvedClassifier(paths, Paths.get("data/sentiment_lexicon")));
	}
}