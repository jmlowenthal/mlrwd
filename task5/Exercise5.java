package mlrwd.task5;

import mlrwd.task2.*;
import mlrwd.task1.*;
import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.stream.*;

public class Exercise5 implements IExercise5 {
	public List<Map<Path, Sentiment>> splitCVRandom(Map<Path, Sentiment> dataSet, int seed) {
		// Prepare folds
		List<Map<Path, Sentiment>> folds = new ArrayList<>(10);
		for (int i = 0; i < 10; ++i) {
			folds.add(new HashMap<Path, Sentiment>());
		}

		// Prep randomised data
		List<Map.Entry<Path, Sentiment>> shuffledData = new ArrayList<Map.Entry<Path, Sentiment>>(dataSet.entrySet());
		Collections.shuffle(shuffledData, new Random(seed));

		// Iterate folds and add the current path
		int i = 0;
		for (Map.Entry<Path, Sentiment> entry : shuffledData) {
			Path path = entry.getKey();
			Sentiment sentiment = entry.getValue();
			folds.get(i).put(path, sentiment);
			i = (i + 1) % folds.size();
		}
		
		return folds;
	}

	public List<Map<Path, Sentiment>> splitCVStratifiedRandom(Map<Path, Sentiment> dataSet, int seed) {
		// Split dataset by sentiment
		Map<Sentiment, List<Path>> sentiments = new HashMap<>();
		for (Sentiment sentiment : Sentiment.values()) {
			List<Path> paths = dataSet.entrySet().stream()
				.filter(e -> e.getValue().equals(sentiment))
				.map(e -> e.getKey())
				.collect(Collectors.toList());
			Collections.shuffle(paths, new Random(seed));
			sentiments.put(sentiment, paths);
		}

		// Prepare folds
		List<Map<Path, Sentiment>> folds = new ArrayList<>(10);
		for (int i = 0; i < 10; ++i) {
			folds.add(new HashMap<Path, Sentiment>());
		}

		// Iterate through the folds and adds the current path
		int i = 0;
		for (Map.Entry<Sentiment, List<Path>> entry : sentiments.entrySet()) {
			Sentiment sentiment = entry.getKey();
			List<Path> paths = entry.getValue();
			for (Path path : paths) {
				folds.get(i).put(path, sentiment);
				i = (i + 1) % folds.size();
			}
		}

		return folds;
	}

	public double[] crossValidate(List<Map<Path, Sentiment>> folds) throws IOException {
		IExercise1 checker = (IExercise1) new Exercise1();

		double[] scores = new double[folds.size()];
		for (int i = 0; i < folds.size(); ++i) {
			// Pick training and testing sets
			Map<Path, Sentiment> testingSet = folds.get(i);
			Map<Path, Sentiment> trainingSet = new HashMap<Path, Sentiment>();
			for (int j = 0; j < folds.size(); ++j) {
				if (i != j) trainingSet.putAll(folds.get(j));
			}

			// Training classifier
			IExercise2 implementation = (IExercise2) new Exercise2();
			Set<Path> testSet = testingSet.keySet();
			Map<String, Map<Sentiment, Double>> tokenLogProbs = implementation.calculateSmoothedLogProbs(trainingSet);
			Map<Sentiment, Double> classProbabilities = implementation.calculateClassProbabilities(trainingSet);

			// Use Naive Bayes
			Map<Path, Sentiment> nb = implementation.naiveBayes(testSet, tokenLogProbs, classProbabilities);
			scores[i] = checker.calculateAccuracy(testingSet, nb);
		}
		return scores;
	}

	public double cvAccuracy(double[] scores) {
		return DoubleStream.of(scores).average().getAsDouble();
	}

	public double cvVariance(double[] scores) {
		double avg = DoubleStream.of(scores).average().getAsDouble();
		return DoubleStream.of(scores).map(d -> (d - avg) * (d - avg)).average().getAsDouble();
	}
}