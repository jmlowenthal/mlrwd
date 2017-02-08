package mlrwd.task6;

import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.util.*;
import java.nio.file.*;

public class Exercise6 implements IExercise6 {
	@Override
	public Map<NuancedSentiment, Double> calculateClassProbabilities(Map<Path, NuancedSentiment> trainingSet) throws IOException {
		Map<NuancedSentiment, Double> map = new HashMap<>();
		for (NuancedSentiment NuancedSentiment : NuancedSentiment.values()) {
			map.put(NuancedSentiment, 0.0);
		}

		double size = trainingSet.size();
		for (NuancedSentiment NuancedSentiment : trainingSet.values()) {
			map.put(NuancedSentiment, map.get(NuancedSentiment) + 1.0 / size);
		}
		return map;		
	}
	
	@Override
	public Map<String, Map<NuancedSentiment, Double>> calculateNuancedLogProbs(Map<Path, NuancedSentiment> trainingSet) throws IOException {
		Map<String, Map<NuancedSentiment, Double>> wordMap = new HashMap<>();
		Map<NuancedSentiment, Integer> sentimentWordCount = new HashMap<>();

		for (NuancedSentiment s : NuancedSentiment.values()) {
			sentimentWordCount.put(s, 0);
		}

		// Iterate training set
		for (Map.Entry<Path, NuancedSentiment> entry : trainingSet.entrySet()) {
			Path path = entry.getKey();
			NuancedSentiment NuancedSentiment = entry.getValue();

			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				// Increment NuancedSentiment word count
				sentimentWordCount.put(NuancedSentiment, sentimentWordCount.get(NuancedSentiment) + 1);

				Map<NuancedSentiment, Double> probs;
				if (wordMap.containsKey(token)) {
					probs = wordMap.get(token);
				}
				else {
					probs = new HashMap<>();
					wordMap.put(token, probs);
					for (NuancedSentiment s : NuancedSentiment.values()) {
						probs.put(s, 1.0);
						sentimentWordCount.put(s, sentimentWordCount.get(s) + 1);
					}
				}

				probs.put(NuancedSentiment, probs.get(NuancedSentiment) + 1.0);
			}
		}

		for (Map<NuancedSentiment, Double> probMap : wordMap.values()) {
			for (Map.Entry<NuancedSentiment, Double> sentimentEntry : probMap.entrySet()) {
				sentimentEntry.setValue(Math.log(sentimentEntry.getValue() / sentimentWordCount.get(sentimentEntry.getKey())));
			}
		}

		return wordMap;
	}

	public static NuancedSentiment argmax(Map<NuancedSentiment, Double> map) {
		Iterator<Map.Entry<NuancedSentiment, Double>> itr = map.entrySet().iterator();

		if (!itr.hasNext()) return null;
		Map.Entry<NuancedSentiment, Double> max = itr.next();
		while (itr.hasNext()) {
			Map.Entry<NuancedSentiment, Double> entry = itr.next();
			if (entry.getValue() > max.getValue()) {
				max = entry;
			}
		}
		return max.getKey();
	}
	
	@Override
	public 	Map<Path, NuancedSentiment> nuancedClassifier(Set<Path> testSet, Map<String, Map<NuancedSentiment, Double>> tokenLogProbs, Map<NuancedSentiment, Double> classProbabilities) throws IOException {
		Map<Path, NuancedSentiment> sentimentMap = new HashMap<>();

		for (Path path : testSet) {
			// Storage for working value of Naive Bayes formula
			Map<NuancedSentiment, Double> bayesClass = new HashMap<>();

			// Iterate tokenised review
			List<String> tokens = Tokenizer.tokenize(path);
			for (String token : tokens) {
				// Fetch log(P(w_i | c))
				Map<NuancedSentiment, Double> logProbs = tokenLogProbs.get(token);
				if (logProbs == null) continue;

				// Iterate different sentiments and add log(P(w_i | c)) to cumulative sum
				for (Map.Entry<NuancedSentiment, Double> entry : logProbs.entrySet()) {
					NuancedSentiment NuancedSentiment = entry.getKey();
					double prob = entry.getValue();

					// Check for NuancedSentiment sum existence in map
					if (bayesClass.containsKey(NuancedSentiment)) {
						bayesClass.put(NuancedSentiment, bayesClass.get(NuancedSentiment) + prob);
					}
					else {
						bayesClass.put(NuancedSentiment, prob);
					}
				}
			}
			
			// Iterate P(c)
			for (Map.Entry<NuancedSentiment, Double> entry : classProbabilities.entrySet()) {
				NuancedSentiment NuancedSentiment = entry.getKey();
				double prob = entry.getValue();

				// Check for existence of NuancedSentiment probability and add log(P(x))
				if (bayesClass.containsKey(NuancedSentiment)) {
					bayesClass.put(NuancedSentiment, bayesClass.get(NuancedSentiment) + Math.log(prob));
				}
				else {
					// This shouldn't happen, hopefully
					// This means that the review contained no words associated with this NuancedSentiment!
					bayesClass.put(NuancedSentiment, Double.NEGATIVE_INFINITY);
				}
			}

			sentimentMap.put(path, argmax(bayesClass));
		}

		return sentimentMap;
	}
	
	@Override
	public double nuancedAccuracy(Map<Path, NuancedSentiment> trueSentiments, Map<Path, NuancedSentiment> predictedSentiments) {
		double correct = 0;
		for (Map.Entry<Path, NuancedSentiment> predicted : predictedSentiments.entrySet()) {
			if (predicted.getValue() == trueSentiments.get(predicted.getKey())) {
				++correct;
			}
		}
		return correct / predictedSentiments.size();
	}

	@Override
	public Map<Integer, Map<Sentiment, Integer>> agreementTable(Collection<Map<Integer, Sentiment>> predictedSentiments) {
		Map<Integer, Map<Sentiment, Integer>> map = new HashMap<>();
		for (Map<Integer, Sentiment> prediction : predictedSentiments) {
			for (Map.Entry<Integer, Sentiment> entry : prediction.entrySet()) {
				Integer review = entry.getKey();
				Sentiment sentiment = entry.getValue();

				Map<Sentiment, Integer> reviewMap = map.get(review);
				if (reviewMap == null) {
					reviewMap = new HashMap<Sentiment, Integer>();
					map.put(review, reviewMap);
				}

				if (reviewMap.containsKey(sentiment)) {
					reviewMap.put(sentiment, reviewMap.get(sentiment) + 1);
				}
				else {
					reviewMap.put(sentiment, 1);
				}
			}
		}
		return map;
	}

	@Override
	public double kappa(Map<Integer, Map<Sentiment, Integer>> agreementTable) {
		Set<Integer> iSet = agreementTable.keySet();
		Map<Integer, Double> n = new HashMap<>();

		for (int i : iSet) {
			double ni = 0.0;
			for (Sentiment s : Sentiment.values()) {
				Map<Sentiment, Integer> map = agreementTable.get(i);
				if (map != null && map.containsKey(s)) {
					ni += map.get(s);
				}
			}
			n.put(i, ni);
		}
		
		double p_e = 0.0;
		for (Sentiment j : Sentiment.values()) {
			double inner = 0.0;
			for (int i : iSet) {
				Map<Sentiment, Integer> map = agreementTable.get(i);
				if (map != null && map.containsKey(j)) {
					inner += map.get(j) / n.get(i);
				}
			}
			inner /= iSet.size();
			p_e += inner * inner;
		}

		double p_a = 0.0;
		for (int i :iSet) {
			double inner = 0.0;
			for (Sentiment j : Sentiment.values()) {
				Map<Sentiment, Integer> map = agreementTable.get(i);
				if (map != null && map.containsKey(j)) {
					double nij = map.get(j);
					inner += nij * (nij - 1);
				}
			}

			double ni = n.get(i);
			p_a += inner / (ni * (ni - 1));
		}
		p_a /= iSet.size();
		
		return (p_a - p_e) / (1.0 - p_e);
	}
}