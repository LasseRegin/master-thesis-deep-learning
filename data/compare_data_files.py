
import os
import json
import math

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

from utils.strings import truncate_sentences, extract_words

dir = os.path.dirname(os.path.abspath(__file__))

data_files = [
    {
        'filename': 'ehealthforumQAs.json',
        'name':     'eHealth forum'
    },
    {
        'filename': 'healthtapQAs.json',
        'name':     'HealthTap'
    },
    {
        'filename': 'icliniqQAs.json',
        'name':     'iCliniq'
    },
    {
        'filename': 'questionDoctorQAs.json',
        'name':     'Question Doctors'
    },
    {
        'filename': 'webmdQA.json',
        'name':     'webMD'
    },
]
file_folder = os.path.join(dir, 'preprocessed-scrape-files')
for data_file in data_files:
    data_file['filename'] = os.path.join(file_folder, data_file['filename'])

tableau20 = [(31, 119, 180),  (174, 199, 232), (255, 127, 14),  (255, 187, 120),
             (44, 160, 44),   (152, 223, 138), (214, 39, 40),   (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75),   (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34),  (219, 219, 141), (23, 190, 207),  (158, 218, 229)]

tableau20_dark = []
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i]        = (r / 255., g / 255., b / 255.)
    tableau20_dark.append((r / 275., g / 275., b / 275.))


def plot_distributions(data_files):
    def remove_outliers(X):
        return X[abs(X - np.mean(X)) < 3 * np.std(X)]

    mus_question, sigmas_question = [], []
    mus_answer,   sigmas_answer   = [], []

    for i, data_file in enumerate(data_files):
        with open(data_file['filename'], 'r') as f:
            _data = json.load(f)

        question_lengths = np.asarray([len(truncate_sentences(x['question'], max_sentences=3)) for x in _data])
        answer_lengths   = np.asarray([len(truncate_sentences(x['answer'], max_sentences=3))   for x in _data])

        question_lengths = remove_outliers(question_lengths)
        answer_lengths   = remove_outliers(answer_lengths)

        # best fit of data
        (mu_answer,     sigma_answer)   = norm.fit(answer_lengths)
        (mu_question,   sigma_question) = norm.fit(question_lengths)
        mus_question.append(mu_question)
        sigmas_question.append(sigma_question)
        mus_answer.append(mu_answer)
        sigmas_answer.append(sigma_answer)

        # axes[i][0].hist(question_lengths)
        # axes[i][1].hist(answer_lengths)
        # axes[0].hist(question_lengths, alpha=0.5, normed=True)
        # n, bins, patches = axes[1].hist(answer_lengths, alpha=0.1, normed=True, color=tableau20[i])

        # y = mlab.normpdf(bins, mu, sigma)
        # l = axes[1].plot(bins, y, '-', linewidth=1, color=tableau20[i])

        # del _data

    #fig, axes = plt.subplots(2, 1)

    two_pi = 2 * math.pi
    def normpdf(x, mean, std):
        variance = std ** 2
        denominator = np.sqrt((two_pi * variance))
        numerator   = np.exp(-(x - mean) ** 2 / (2 * variance))
        return numerator / denominator

    x_question = np.arange(0, 400)
    x_answer   = np.arange(0, 700)

    fig = plt.figure(figsize=(10,3))
    for i, data_file in enumerate(data_files):
        plt.plot(x_question, normpdf(x_question, mean=mus_question[i], std=sigmas_question[i]), color=tableau20[i], label=data_file['name'], linewidth=2)
    plt.title('Questions')
    plt.yticks([])
    plt.legend()
    fig.savefig('question_seq_length_comparison.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(10,3))
    for i, data_file in enumerate(data_files):
        plt.plot(x_answer, normpdf(x_answer, mean=mus_answer[i], std=sigmas_answer[i]), color=tableau20[i], label=data_file['name'], linewidth=2)
    plt.title('Answers')
    plt.yticks([])
    plt.legend()
    fig.savefig('answer_seq_length_comparison.pdf', bbox_inches='tight')



    # for i, data_file in enumerate(data_files):
    #     plt.plot(x_answer,   normpdf(x_answer,   mean=mus_answer[i],   std=sigmas_answer[i]),   color=tableau20[i], label=data_file['name'], linewidth=2)

    # # plt.legend()
    # axes[0].set_title('Questions')
    # axes[0].yaxis.set_ticks([])
    # axes[0].legend()

    # axes[1].yaxis.set_ticks([])
    # axes[1].set_title('Answers')
    # axes[1].legend()
    # fig.savefig('qa_seq_length_comparison.pdf', bbox_inches='tight')
    # plt.show()


def plot_pca(data_files):
    vectorizer = CountVectorizer(stop_words='english', min_df=10, max_df=0.50)
    #vectorizer = TfidfVectorizer(stop_words='english', min_df=10, max_df=0.50)

    data_files = list(reversed(data_files))

    documents = []
    for i, data_file in enumerate(data_files):
        with open(data_file['filename'], 'r') as f:
            _data = json.load(f)
            for x in _data:
                for string in [x['question'], x['answer']]:
                    documents.append(
                        ' '.join([word for word in extract_words(string)])
                    )

    vectorizer.fit(documents)


    X, y = [], []
    for i, data_file in enumerate(data_files):
        with open(data_file['filename'], 'r') as f:
            _data = json.load(f)

        for x in _data:
            question = x['question']
            answer = x['answer']

            X.append(vectorizer.transform(['%s %s' % (question, answer)])[0].toarray().squeeze())
            y.append(i)
    X = np.asarray(X)
    y = np.asarray(y)

    # Project on pca components
    pca = PCA(n_components=3)
    projections = pca.fit_transform(X)

    fig = plt.figure()
    for i, data_file in enumerate(data_files):
        plt.scatter(projections[y == i, 0], projections[y == i, 1], color=tableau20[i], label=data_file['name'])
    plt.legend()
    plt.title('Explained variance = %.4f' % (pca.explained_variance_ratio_[0:2].sum()))
    plt.show()


if __name__ == '__main__':
    plot_distributions(data_files)
    # plot_pca(data_files)
