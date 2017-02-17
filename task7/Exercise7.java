package mlrwd.task7;

import uk.ac.cam.cl.mlrwd.exercises.markov_models.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class Exercise7 implements IExercise7 {
	public HiddenMarkovModel<DiceRoll, DiceType> estimateHMM(Collection<Path> sequenceFiles) throws IOException {
		
		// Initialise transitional matrix and metadata
		int transitionalCount = 0;
		Map<DiceType, Map<DiceType, Double>> transitionalMatrix = new HashMap<>();
		for (DiceType a : DiceType.values()) {
			Map<DiceType, Double> map = new HashMap<>();
			transitionalMatrix.put(a, map);
			for (DiceType b : DiceType.values()) {
				map.put(b, 0.0);
			}
		}

		// Initialise emission matrix and metadata
		Map<DiceType, Integer> emissionCount = new HashMap<>();
		Map<DiceType, Map<DiceRoll, Double>> emissionMatrix = new HashMap<>();
		for (DiceType type : DiceType.values()) {
			emissionCount.put(type, 0);
			Map<DiceRoll, Double> map = new HashMap<>();
			emissionMatrix.put(type, map);
			for (DiceRoll roll : DiceRoll.values()) {
				map.put(roll, 0.0);
			}
		}
		
		Map<DiceType, Double> initialProbs = new HashMap<>();
		for (DiceType type : DiceType.values()) {
			initialProbs.put(type, 0.0);
		}

		// Iterate data
		List<HMMDataStore<DiceRoll, DiceType>> stores = HMMDataStore.loadDiceFiles(sequenceFiles);
		for (HMMDataStore<DiceRoll, DiceType> store : stores) {
			// Transitional calcuations
			List<DiceType> hidden = store.hiddenSequence;
			for (int i = 1; i < hidden.size(); ++i) {
				DiceType initial = hidden.get(i - 1);
				DiceType result = hidden.get(i);
				Map<DiceType, Double> map = transitionalMatrix.get(initial);
				map.put(result, map.get(result) + 1.0);
				++transitionalCount;
			}

			// Emission calculations
			List<DiceRoll> observed = store.observedSequence;
			for (int i = 0; i < observed.size() && i < hidden.size(); ++i) {
				DiceType type = hidden.get(i);
				DiceRoll roll = observed.get(i);
				Map<DiceRoll, Double> map = emissionMatrix.get(type);
				map.put(roll, map.get(roll) + 1.0);
				emissionCount.put(type, emissionCount.get(type) + 1);
			}

			// Initial probability calculations
			DiceType startState = hidden.get(0);
			initialProbs.put(startState, initialProbs.get(startState) + 1);
		}

		// Divide by transitional count
		for (Map<DiceType, Double> map : transitionalMatrix.values()) {
			for (Map.Entry<DiceType, Double> entry : map.entrySet()) {
				entry.setValue(entry.getValue() / transitionalCount);
			}
		}

		// Divide by emission counts
		for (Map.Entry<DiceType, Map<DiceRoll, Double>> outer : emissionMatrix.entrySet()) {
			int count = emissionCount.get(outer.getKey());
			Map<DiceRoll, Double> map = outer.getValue();
			for (Map.Entry<DiceRoll, Double> entry : map.entrySet()) {
				entry.setValue(entry.getValue() / count);
			}
		}

		// Divide by sequence count
		for (Map.Entry<DiceType, Double> entry : initialProbs.entrySet()) {
			entry.setValue(entry.getValue() / stores.size());
		}

		return new HiddenMarkovModel<DiceRoll, DiceType>(transitionalMatrix, emissionMatrix, initialProbs);
	}
}