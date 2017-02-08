package mlrwd.task2;

import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Exercise2 implements IExercise2 {
	
	public Map<Sentiment, Double> calculateClassProbabilities(
		Map<Path, Sentiment> trainingSet
	) throws IOException {
		Map<Sentiment, Double> map = new HashMap<>();
		for (Sentiment s : Sentiment.values()) {
			if (map.containsKey(s)) {
				map.put(s, map.get(s) + 1.0);
			}
			else {
				map.put(s, 1.0);
			}
		}

		int size = trainingSet.size();
		for (Sentiment s : map.keySet()) {
			map.put(s, map.get(s) / size);
		}

		return map;
	}

	public Map<String, Map<Sentiment, Double>> calculateUnsmoothedLogProbs(
			Map<Path, Sentiment> trainingSet
	) throws IOException {
		Map<String, Map<Sentiment, Double>> wordMap = new HashMap<>();
		Map<Sentiment, Integer> sentimentWordCount = new HashMap<>();

		// Iterate training set
		for (Map.Entry<Path, Sentiment> entry : trainingSet.entrySet()) {
			Path path = entry.getKey();
			Sentiment sentiment = entry.getValue();

			if (!sentimentWordCount.containsKey(sentiment)) {
				sentimentWordCount.put(sentiment, 0);
			}

			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				// Increment sentiment word count
				sentimentWordCount.put(sentiment, sentimentWordCount.get(sentiment) + 1);

				Map<Sentiment, Double> probs;
				if (wordMap.containsKey(token)) {
					probs = wordMap.get(token);
				}
				else {
					probs = new HashMap<>();
					wordMap.put(token, probs);
					probs.put(Sentiment.POSITIVE, 0.0);
					probs.put(Sentiment.NEGATIVE, 0.0);
				}

				probs.put(sentiment, probs.get(sentiment) + 1.0);
			}
		}

		for (Map<Sentiment, Double> probMap : wordMap.values()) {
			for (Map.Entry<Sentiment, Double> sentimentEntry : probMap.entrySet()) {
				sentimentEntry.setValue(Math.log(sentimentEntry.getValue() / sentimentWordCount.get(sentimentEntry.getKey())));
			}
		}

		return wordMap;
	}

	public Map<String, Map<Sentiment, Double>> calculateSmoothedLogProbs(
		Map<Path, Sentiment> trainingSet
	) throws IOException {
		Map<String, Map<Sentiment, Double>> wordMap = new HashMap<>();
		Map<Sentiment, Integer> sentimentWordCount = new HashMap<>();

		for (Sentiment s : Sentiment.values()) {
			sentimentWordCount.put(s, 0);
		}

		// Iterate training set
		for (Map.Entry<Path, Sentiment> entry : trainingSet.entrySet()) {
			Path path = entry.getKey();
			Sentiment sentiment = entry.getValue();

			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				// Increment sentiment word count
				sentimentWordCount.put(sentiment, sentimentWordCount.get(sentiment) + 1);

				Map<Sentiment, Double> probs;
				if (wordMap.containsKey(token)) {
					probs = wordMap.get(token);
				}
				else {
					probs = new HashMap<>();
					wordMap.put(token, probs);
					for (Sentiment s : Sentiment.values()) {
						probs.put(s, 1.0);
						sentimentWordCount.put(s, sentimentWordCount.get(s) + 1);
					}
				}

				probs.put(sentiment, probs.get(sentiment) + 1.0);
			}
		}

		for (Map<Sentiment, Double> probMap : wordMap.values()) {
			for (Map.Entry<Sentiment, Double> sentimentEntry : probMap.entrySet()) {
				sentimentEntry.setValue(Math.log(sentimentEntry.getValue() / sentimentWordCount.get(sentimentEntry.getKey())));
			}
		}

		return wordMap;
	}

	public static Sentiment argmax(Map<Sentiment, Double> map) {
		Iterator<Map.Entry<Sentiment, Double>> itr = map.entrySet().iterator();

		if (!itr.hasNext()) return null;
		Map.Entry<Sentiment, Double> max = itr.next();
		while (itr.hasNext()) {
			Map.Entry<Sentiment, Double> entry = itr.next();
			if (entry.getValue() > max.getValue()) {
				max = entry;
			}
		}
		return max.getKey();
	}

	public Map<Path, Sentiment> naiveBayes(
		Set<Path> testSet,
		Map<String, Map<Sentiment, Double>> tokenLogProbs,
		Map<Sentiment, Double> classProbabilities
	) throws IOException {
		Map<Path, Sentiment> sentimentMap = new HashMap<>();

		for (Path path : testSet) {
			// Storage for working value of Naive Bayes formula
			Map<Sentiment, Double> bayesClass = new HashMap<>();

			// Iterate tokenised review
			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				// Fetch log(P(w_i | c))
				Map<Sentiment, Double> logProbs = tokenLogProbs.get(token);
				if (logProbs == null) continue;

				// Iterate different sentiments and add log(P(w_i | c)) to cumulative sum
				for (Map.Entry<Sentiment, Double> entry : logProbs.entrySet()) {
					Sentiment sentiment = entry.getKey();
					double prob = entry.getValue();

					// Check for sentiment sum existence in map
					if (bayesClass.containsKey(sentiment)) {
						bayesClass.put(sentiment, bayesClass.get(sentiment) + prob);
					}
					else {
						bayesClass.put(sentiment, prob);
					}
				}
			}
			
			// Iterate P(c)
			for (Map.Entry<Sentiment, Double> entry : classProbabilities.entrySet()) {
				Sentiment sentiment = entry.getKey();
				double prob = entry.getValue();

				// Check for existence of sentiment probability and add log(P(x))
				if (bayesClass.containsKey(sentiment)) {
					bayesClass.put(sentiment, bayesClass.get(sentiment) + Math.log(prob));
				}
				else {
					// This shouldn't happen, hopefully
					// This means that the review contained no words associated with this sentiment!
					bayesClass.put(sentiment, Double.NEGATIVE_INFINITY);
				}
			}

			sentimentMap.put(path, argmax(bayesClass));
		}

		return sentimentMap;
	}

	public static void main(String[] args) throws IOException {
		Exercise2 ex2 = new Exercise2();

		Set<Path> paths = new HashSet<Path>();
		paths.add(Paths.get("/home/jamie/wiki/compsci/tasks/mlrwd1 - opinion1.txt"));
		
		Path dataDirectory = Paths.get("data/sentiment_dataset");
		Map<Path, Sentiment> dataSet = DataPreparation1.loadSentimentDataset(dataDirectory.resolve("reviews"), dataDirectory.resolve("review_sentiment"));

		Map<Sentiment, Double> classProbabilities = ex2.calculateClassProbabilities(dataSet);

		// Unsmoothed
		Map<String, Map<Sentiment, Double>> logProbs = ex2.calculateUnsmoothedLogProbs(dataSet);

		Map<Path, Sentiment> NBPredictions = ex2.naiveBayes(paths, logProbs,
				classProbabilities);

		// Smoothed
		Map<String, Map<Sentiment, Double>> smoothedLogProbs = ex2
				.calculateSmoothedLogProbs(dataSet);

		// Naive Bayes
		Map<Path, Sentiment> smoothedNBPredictions = ex2.naiveBayes(paths,
				smoothedLogProbs, classProbabilities);

		System.out.println(NBPredictions);
		System.out.println(smoothedNBPredictions);
	}
}
