package mlrwd.task9;

import java.io.IOException;
import java.util.*;

import uk.ac.cam.cl.mlrwd.exercises.markov_models.*;

public class Exercise9 implements IExercise9 {
	public Map<Feature, Double> getFinalProbs(List<HMMDataStore<AminoAcid, Feature>> trainingPairs) throws IOException {
		Map<Feature, Double> endProbs = new HashMap<>();
		for (Feature type : Feature.values()) {
			endProbs.put(type, 0.0);
		}

		// Iterate data
		double total = 0.0;
		for (HMMDataStore<AminoAcid, Feature> store : trainingPairs) {
			// Final probability calculations
			List<Feature> hidden = store.hiddenSequence;
			Feature endState = hidden.get(hidden.size() - 1);
			endProbs.put(endState, endProbs.get(endState) + 1);
			++total;
		}

		for (Feature type : Feature.values()) {
			endProbs.put(type, endProbs.get(type) / total);
			if (endProbs.get(type) == 0.0) endProbs.remove(type);
		}

		return endProbs;
	}

	public HiddenMarkovModel<AminoAcid, Feature> estimateHMM(List<HMMDataStore<AminoAcid, Feature>> sequencePairs) throws IOException {
		
		// Initialise transitional matrix and metadata
		int transitionalCount = 0;
		Map<Feature, Map<Feature, Double>> transitionalMatrix = new HashMap<>();
		for (Feature a : Feature.values()) {
			Map<Feature, Double> map = new HashMap<>();
			transitionalMatrix.put(a, map);
			for (Feature b : Feature.values()) {
				map.put(b, 0.0);
			}
		}

		// Initialise emission matrix and metadata
		Map<Feature, Integer> emissionCount = new HashMap<>();
		Map<Feature, Map<AminoAcid, Double>> emissionMatrix = new HashMap<>();
		for (Feature type : Feature.values()) {
			emissionCount.put(type, 0);
			Map<AminoAcid, Double> map = new HashMap<>();
			emissionMatrix.put(type, map);
			for (AminoAcid roll : AminoAcid.values()) {
				map.put(roll, 0.0);
			}
		}
		
		Map<Feature, Double> initialProbs = new HashMap<>();
		for (Feature type : Feature.values()) {
			initialProbs.put(type, 0.0);
		}

		// Iterate data
		for (HMMDataStore<AminoAcid, Feature> store : sequencePairs) {
			// Transitional calcuations
			List<Feature> hidden = store.hiddenSequence;
			for (int i = 1; i < hidden.size(); ++i) {
				Feature initial = hidden.get(i - 1);
				Feature result = hidden.get(i);
				Map<Feature, Double> map = transitionalMatrix.get(initial);
				map.put(result, map.get(result) + 1.0);
				++transitionalCount;
			}

			// Emission calculations
			List<AminoAcid> observed = store.observedSequence;
			for (int i = 0; i < observed.size() && i < hidden.size(); ++i) {
				Feature type = hidden.get(i);
				AminoAcid roll = observed.get(i);
				Map<AminoAcid, Double> map = emissionMatrix.get(type);
				map.put(roll, map.get(roll) + 1.0);
				emissionCount.put(type, emissionCount.get(type) + 1);
			}

			// Initial probability calculations
			Feature startState = hidden.get(0);
			initialProbs.put(startState, initialProbs.get(startState) + 1);
		}

		// Divide by transitional count
		for (Map<Feature, Double> map : transitionalMatrix.values()) {
			for (Map.Entry<Feature, Double> entry : map.entrySet()) {
				entry.setValue(entry.getValue() / transitionalCount);
			}
		}

		// Divide by emission counts
		for (Map.Entry<Feature, Map<AminoAcid, Double>> outer : emissionMatrix.entrySet()) {
			int count = emissionCount.get(outer.getKey());
			Map<AminoAcid, Double> map = outer.getValue();
			for (Map.Entry<AminoAcid, Double> entry : map.entrySet()) {
				entry.setValue(entry.getValue() / count);
			}
		}

		// Divide by sequence count
		for (Map.Entry<Feature, Double> entry : initialProbs.entrySet()) {
			entry.setValue(entry.getValue() / sequencePairs.size());
		}

		return new HiddenMarkovModel<AminoAcid, Feature>(transitionalMatrix, emissionMatrix, initialProbs);

	}

	public List<Feature> viterbi(HiddenMarkovModel<AminoAcid, Feature> model, Map<Feature, Double> finalProbs, List<AminoAcid> observedSequence) {
		List<Map<Feature, Feature>> backtrackPath = new ArrayList<>(); // phi
		List<Map<Feature, Double>> pathProbability = new ArrayList<>(); // delta

		Feature finalState = null;
		Double logPredictionProb = Double.NEGATIVE_INFINITY;
		
		// Initial starting state
		Map<Feature, Double> initialMap = new HashMap<>();
		Map<Feature, Double> a = model.getInitialProbs();
		for (Feature type : Feature.values()) {
			Double a0 = a.get(type);
			//Map<AminoAcid, Double> b = model.getPossibleEmissions(type);
			//Double b0 = b.get(observedSequence.get(0)); // Factored in later
			
			if (a0 == null/* || b0 == null*/) continue;

			initialMap.put(type, Math.log(a0)/* + Math.log(b0)*/);
		}
		pathProbability.add(initialMap);

		// The rest
		// delta_j(t + 1) = log(b_j(o_{t + 1})) + max(log(delta_i(t) * a_{ij}))
		for (int t = 0; t <= observedSequence.size(); ++t) {
			Map<Feature, Double> currentMap = pathProbability.get(t);
			Map<Feature, Double> nextMap = new HashMap<>();
			pathProbability.add(nextMap); // Map for the next time-step

			backtrackPath.add(new HashMap<>());

			if (t == observedSequence.size()) {
				// This is the penultimate state, calculate singular final state
				pathProbability.remove(t + 1); // get rid of map, we have a singular variable
				for (Map.Entry<Feature, Double> entry : finalProbs.entrySet()) {
					Feature typei = entry.getKey();
					Double log_aif = Math.log(entry.getValue());
					Double log_deltai = currentMap.get(typei);
					
					if (log_deltai + log_aif > logPredictionProb) {
						finalState = typei;
						logPredictionProb = log_deltai + log_aif;
						backtrackPath.get(t).put(finalState, typei);
					}
				}
			}
			else {
				for (Feature typei : Feature.values()) { // Loop current steps
					// Factor in emission probability -- log(b_j(o_t))
					Double bj = model.getPossibleEmissions(typei).get(observedSequence.get(t));
					Double log_deltai = currentMap.getOrDefault(typei, 0.0) + Math.log(bj);
					currentMap.put(typei, log_deltai);

					// Look ahead an project if maximum -- max(log(delta_i(t)) + log(a_{ij})) 
					Map<Feature, Double> transitions = model.getPossibleTransitions(typei);
					for (Map.Entry<Feature, Double> entry : transitions.entrySet()) {
						Feature typej = entry.getKey();
						Double log_aij = Math.log(entry.getValue());

						// Check nextMap has entry
						if (nextMap.containsKey(typej)) {
							// Check if current path is better
							Double log_deltaj = nextMap.get(typej);
							if (log_deltai + log_aij > log_deltaj) {
								nextMap.put(typej, log_deltai + log_aij);
								backtrackPath.get(t).put(typej, typei);
							}
						}
						else {
							// Current path is best (for now...)
							nextMap.put(typej, log_deltai + log_aij);
							backtrackPath.get(t).put(typej, typei);
						}
					}
				}
			}
		}

		// Backtrack
		List<Feature> path = new ArrayList<>();
		Feature currentState = finalState;
		path.add(currentState);
		for (int t = observedSequence.size() - 2; t >= 0; --t) {
			currentState = backtrackPath.get(t).get(currentState);
			path.add(currentState);
		}
		Collections.reverse(path);
		return path;
	}

	public Map<List<Feature>, List<Feature>> predictAll(HiddenMarkovModel<AminoAcid, Feature> model, Map<Feature, Double> finalProbs, List<HMMDataStore<AminoAcid, Feature>> testSequencePairs) throws IOException {
		Map<List<Feature>, List<Feature>> map = new HashMap<>();
		for (HMMDataStore<AminoAcid, Feature> store : testSequencePairs) {
			List<Feature> predicted = viterbi(model, finalProbs, store.observedSequence);
			map.put(store.hiddenSequence, predicted);
		}
		return map;
	}

	public double precision(Map<List<Feature>, List<Feature>> true2PredictedMap) {
		double correct = 0;
		double total = 0;
		for (Map.Entry<List<Feature>, List<Feature>> entry : true2PredictedMap.entrySet()) {
			List<Feature> correctState = entry.getKey();
			List<Feature> predicted = entry.getValue();
			for (int i = 0; i < correctState.size() && i < predicted.size(); ++i) {
				if (predicted.get(i).equals(Feature.MEMBRANE)) {
					++total;
					if (correctState.get(i).equals(predicted.get(i))) {
						++correct;
					}
				}
			}
		}
		return correct / total;
	}
	
	public double recall(Map<List<Feature>, List<Feature>> true2PredictedMap) {
		double correct = 0;
		double total = 0;
		for (Map.Entry<List<Feature>, List<Feature>> entry : true2PredictedMap.entrySet()) {
			List<Feature> correctState = entry.getKey();
			List<Feature> predicted = entry.getValue();
			for (int i = 0; i < correctState.size() && i < predicted.size(); ++i) {
				if (correctState.get(i).equals(Feature.MEMBRANE)) {
					++total;
					if (correctState.get(i).equals(predicted.get(i))) {
						++correct;
					}
				}
			}
		}
		return correct / total;
	}
	
	public double fOneMeasure(Map<List<Feature>, List<Feature>> true2PredictedMap) {
		double p = precision(true2PredictedMap);
		double r = recall(true2PredictedMap);
		return 2 * p * r / (p + r);
	}
}