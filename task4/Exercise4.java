package mlrwd.task4;

import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.math.BigInteger;

public class Exercise4 implements IExercise4 {
    public Map<String, WeightedSentiment> parseLexicon(Path lexiconFile) throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(lexiconFile)) {
			Map<String, WeightedSentiment> lexicon = new HashMap<>();
			String line;
			while ((line = reader.readLine()) != null) {
				Scanner scanner = new Scanner(line);;
				String word = null;
				int strength = 1;
				while (scanner.hasNext()) {
					String pair = scanner.next();
					String[] arr = pair.split("=");
					if (arr.length == 2) {
						switch (arr[0]) {
							case "type": {
								if (arr[1].length() >= 5) {
									if (arr[1].substring(0, 6).equals("strong")) {
										strength = 2;
									}
								}
								break;
							}
							case "word1": {
								word = arr[1];
								break;
							}
							case "priorpolarity": {
								switch (arr[1]) {
									case "positive":
										lexicon.put(word, new WeightedSentiment(Sentiment.POSITIVE, strength));
										break;
									case "negative":
										lexicon.put(word, new WeightedSentiment(Sentiment.NEGATIVE, strength));
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

    @Override
    public Map<Path, Sentiment> magnitudeClassifier(Set<Path> testSet, Path lexiconFile) throws IOException {
        Map<String, WeightedSentiment> lexicon = parseLexicon(lexiconFile);
		Map<Path, Sentiment> result = new HashMap<>();
        
		for (Path path : testSet) {
            int positive = 0, negative = 0;
            List<String> tokens = Tokenizer.tokenize(path);
            for (String token : tokens) {
                WeightedSentiment s = lexicon.get(token);
                if (s != null) {
                    switch (s.getSentiment()) {
                        case POSITIVE:
                            positive += s.getStrength();
                            break;
                        case NEGATIVE:
                            negative += s.getStrength();
                            break;
                    }
                }
            }

			// Positive words are often used in negative reviews (sarcasm) and certainly more
			// common that negative words in positive reviews
			Sentiment predict = positive < negative ? Sentiment.NEGATIVE : Sentiment.POSITIVE;
			result.put(path, predict);
		}
		return result;
    }

    @Override
    public double signTest(Map<Path, Sentiment> actualSentiments, Map<Path, Sentiment> classA,
			Map<Path, Sentiment> classB) {
        int plus = 0, minus = 0, nill = 0;
        for (Map.Entry<Path, Sentiment> entry : actualSentiments.entrySet()) {
            Path path = entry.getKey();
            Sentiment sentiment = entry.getValue();

            Sentiment a = classA.get(path);
            Sentiment b = classB.get(path);
            if (a.equals(b)) {
                ++nill;
            }
            else if (a.equals(sentiment)) {
                ++plus;
            }
            else if (b.equals(sentiment)) {
                ++minus;
            }
            else {
                // Probably not...
                ++nill;
            }
        }

        int n = 2 * (int)Math.ceil(nill / 2.0) + plus + minus;
        int k = (int)Math.ceil(nill / 2.0) + Math.min(plus, minus);
        double q = 0.5;

        double pValue = 0.0;
        for (int i = 0; i <= k; ++i) {
            pValue += choose(n, i) * Math.pow(q, i) * Math.pow(1 - q, n - i);
        }
        return pValue * 2;
    }

    public static double choose(int n, int k) {
		// n! / ((n - k)! k!)
        return factorial(n).divide(factorial(n - k).multiply(factorial(k))).doubleValue();
    }

    public static BigInteger factorial(Integer n) {
        if (n < 1) return BigInteger.ONE;
        return factorial(n - 1).multiply(new BigInteger(n.toString()));
    }
}
