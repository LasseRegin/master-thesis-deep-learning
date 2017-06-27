
import numpy as np


def optimal_candidate_idx(candidates, alphabet, vectorizer, classifier, input_word_features):
    # Construct candidate word features
    candidate_word_features = []
    for candidate in candidates:
        candidate_str = alphabet.seq2str(candidate, decode=True)

        feature_vec = vectorizer.transform([candidate_str]).toarray().squeeze()
        candidate_word_features.append(feature_vec)

    # Predict question category probabilities
    question_probs = classifier.predict_question(input_word_features)

    candidate_word_features = np.asarray(candidate_word_features)

    # Compute candidates category probabilities
    candidates_probs = classifier.predict_answer(candidate_word_features)

    # Chose optimal candidate based on euclidian distance between
    # probability vector of question and candidate.
    dists = []
    for candidate_prob in candidates_probs:
        dist = np.mean((question_probs - candidate_prob) ** 2)
        dists.append(dist)

    return np.argmin(dists)
