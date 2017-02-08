package mlrwd.task5;

import mlrwd.task2.*;
import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Exercise5Star {
	public static void main(String[] args) throws IOException {
		Path dir_2004 = Paths.get("data/sentiment_dataset");
		Path dir_2016 = Paths.get("data/year_2016_dataset");

		Map<String, Map<Sentiment, Double>> probs_2004 = logProbs(dir_2004);
		Map<String, Map<Sentiment, Double>> probs_2016 = logProbs(dir_2016);

		for (String word : probs_2004.keySet()) {
			Map<Sentiment, Double> wordprob_2004 = probs_2004.get(word);
			Map<Sentiment, Double> wordprob_2016 = probs_2016.get(word);

			if (wordprob_2016 == null) continue;

			try {
				double posA = Math.exp(wordprob_2004.get(Sentiment.POSITIVE));
				double negA = Math.exp(wordprob_2004.get(Sentiment.NEGATIVE));
				double posB = Math.exp(wordprob_2016.get(Sentiment.POSITIVE));
				double negB = Math.exp(wordprob_2016.get(Sentiment.NEGATIVE));

				//System.out.printf("A = {%f %f}, B = {%f %f}%n", posA, negA, posB, negB);

				// Gives numerical representation of sentiment in interval [-1 : 1]
				double sentimentA = posA - negA;
				double sentimentB = posB - negB;

				double change = (sentimentB - sentimentA) / (posA + negA + posB + negB);

				if (Math.abs(change) < 0.01) {
					System.out.printf("%s\t\t%.3f%n", word, change);
				}
			}
			catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}
	}

	public static Map<String, Map<Sentiment, Double>> logProbs(Path dir) throws IOException {
		IExercise2 implmentation = (IExercise2) new Exercise2();

		Path sentimentFile = dir.resolve("review_sentiment");
		Map<Path, Sentiment> dataSet = DataPreparation1.loadSentimentDataset(
			dir.resolve("reviews"), dir.resolve("review_sentiment")
		);

		return implmentation.calculateSmoothedLogProbs(dataSet);
	}
}