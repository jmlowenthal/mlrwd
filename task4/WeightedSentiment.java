package mlrwd.task4;

import uk.ac.cam.cl.mlrwd.exercises.sentiment_detection.Sentiment;

public class WeightedSentiment {
    private Sentiment sentiment;
    private int strength = 1;

    public WeightedSentiment(Sentiment sent, int stren) {
        sentiment = sent;
        strength = stren;
    }

    public Sentiment getSentiment() {
        return sentiment;
    }

    public int getStrength() {
        return strength;
    }
        
    @Override
    public String toString() {
        return "{" + getSentiment() + " " + getStrength() + "}";
    }
}
