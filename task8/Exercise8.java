package mlrwd.task8;

import uk.ac.cam.cl.mlrwd.exercises.markov_models.*;

import java.io.IOException;
import java.util.*;
import java.nio.file.Path;

public class Exercise8 implements IExercise8 {
	public Map<DiceType, Double> getFinalProbs(List<Path> trainingFiles) throws IOException {
		Map<DiceType, Double> endProbs = new HashMap<>();
		for (DiceType type : DiceType.values()) {
			endProbs.put(type, 0.0);
		}

		// Iterate data
		double total = 0.0;
		List<HMMDataStore<DiceRoll, DiceType>> stores = HMMDataStore.loadDiceFiles(trainingFiles);
		for (HMMDataStore<DiceRoll, DiceType> store : stores) {
			// Final probability calculations
			List<DiceType> hidden = store.hiddenSequence;
			DiceType endState = hidden.get(hidden.size() - 1);
			endProbs.put(endState, endProbs.get(endState) + 1);
			++total;
		}

		for (DiceType type : DiceType.values()) {
			endProbs.put(type, endProbs.get(type) / total);
		}

		return endProbs;
	}

	public List<DiceType> viterbi(HiddenMarkovModel<DiceRoll, DiceType> model, Map<DiceType, Double> finalProbs, List<DiceRoll> observedSequence) {
		List<Map<DiceType, DiceType>> backtrackPath = new ArrayList<>(); // phi
		List<Map<DiceType, Double>> pathProbability = new ArrayList<>(); // delta

		DiceType finalState = null;
		Double logPredictionProb = Double.NEGATIVE_INFINITY;
		
		// Initial starting state
		Map<DiceType, Double> initialMap = new HashMap<>();
		Map<DiceType, Double> a = model.getInitialProbs();
		for (DiceType type : DiceType.values()) {
			Double a0 = a.get(type);
			//Map<DiceRoll, Double> b = model.getPossibleEmissions(type);
			//Double b0 = b.get(observedSequence.get(0)); // Factored in later
			
			if (a0 == null/* || b0 == null*/) continue;

			initialMap.put(type, Math.log(a0)/* + Math.log(b0)*/);
		}
		pathProbability.add(initialMap);

		// The rest
		// delta_j(t + 1) = log(b_j(o_{t + 1})) + max(log(delta_i(t) * a_{ij}))
		for (int t = 0; t <= observedSequence.size(); ++t) {
			Map<DiceType, Double> currentMap = pathProbability.get(t);
			Map<DiceType, Double> nextMap = new HashMap<>();
			pathProbability.add(nextMap); // Map for the next time-step

			backtrackPath.add(new HashMap<>());

			if (t == observedSequence.size()) {
				// This is the penultimate state, calculate singular final state
				pathProbability.remove(t + 1); // get rid of map, we have a singular variable
				for (Map.Entry<DiceType, Double> entry : finalProbs.entrySet()) {
					DiceType typei = entry.getKey();
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
				for (DiceType typei : DiceType.values()) { // Loop current steps
					// Factor in emission probability -- log(b_j(o_t))
					Double bj = model.getPossibleEmissions(typei).get(observedSequence.get(t));
					Double log_deltai = currentMap.get(typei) + Math.log(bj);
					currentMap.put(typei, log_deltai);

					// Look ahead an project if maximum -- max(log(delta_i(t)) + log(a_{ij})) 
					Map<DiceType, Double> transitions = model.getPossibleTransitions(typei);
					for (Map.Entry<DiceType, Double> entry : transitions.entrySet()) {
						DiceType typej = entry.getKey();
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
		List<DiceType> path = new ArrayList<>();
		DiceType currentState = finalState;
		path.add(currentState);
		for (int t = observedSequence.size() - 2; t >= 0; --t) {
			currentState = backtrackPath.get(t).get(currentState);
			path.add(currentState);
		}
		Collections.reverse(path);
		return path;
	}

	public Map<List<DiceType>, List<DiceType>> predictAll(HiddenMarkovModel<DiceRoll, DiceType> model, Map<DiceType, Double> finalProbs, List<Path> testFiles) throws IOException {
		Map<List<DiceType>, List<DiceType>> map = new HashMap<>();
		List<HMMDataStore<DiceRoll, DiceType>> stores = HMMDataStore.loadDiceFiles(testFiles);
		for (HMMDataStore<DiceRoll, DiceType> store : stores) {
			List<DiceType> predicted = viterbi(model, finalProbs, store.observedSequence);
			map.put(store.hiddenSequence, predicted);
		}
		return map;
	}

	public double precision(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
		double correct = 0;
		double total = 0;
		for (Map.Entry<List<DiceType>, List<DiceType>> entry : true2PredictedMap.entrySet()) {
			List<DiceType> correctState = entry.getKey();
			List<DiceType> predicted = entry.getValue();
			for (int i = 0; i < correctState.size() && i < predicted.size(); ++i) {
				if (predicted.get(i).equals(DiceType.FAIR)) continue;
				++total;
				if (correctState.get(i).equals(predicted.get(i))) {
					++correct;
				}
			}
		}
		return correct / total;
	}

	public double recall(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
		double correct = 0;
		double total = 0;
		for (Map.Entry<List<DiceType>, List<DiceType>> entry : true2PredictedMap.entrySet()) {
			List<DiceType> correctState = entry.getKey();
			List<DiceType> predicted = entry.getValue();
			for (int i = 0; i < correctState.size() && i < predicted.size(); ++i) {
				if (correctState.get(i).equals(DiceType.FAIR)) continue;
				++total;
				if (correctState.get(i).equals(predicted.get(i))) {
					++correct;
				}
			}
		}
		return correct / total;
	}
	
	public double fOneMeasure(Map<List<DiceType>, List<DiceType>> true2PredictedMap) {
		double p = precision(true2PredictedMap);
		double r = recall(true2PredictedMap);
		return 2 * p * r / (p + r);
	}
}