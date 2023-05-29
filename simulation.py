import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from utils.plots import plot_popularities_over_time

n_parties = 10
n_voters = 10_000
n_elections = 100


def utility(
    voter_beliefs: np.ndarray,
    party_beliefs: np.ndarray,
    party_popularity: np.ndarray,
    tactical_bias: float = 0.5,
) -> np.ndarray:
    distances = euclidean_distances(voter_beliefs, party_beliefs)
    affinities = 1 / distances
    absolute_utility = party_popularity * affinities
    return absolute_utility


def get_popularity(votes: np.ndarray, n_parties: int) -> np.ndarray:
    uniques, counts = np.unique(votes, return_counts=True)
    popularity = np.zeros(n_parties)
    for party, count in zip(uniques, counts):
        popularity[party] = count
    popularity = popularity / np.sum(popularity)
    return popularity


n_parties = 10
n_voters = 10_000
n_elections = 100

voter_beliefs = np.random.normal(loc=0.0, scale=1.0, size=n_voters)
voter_beliefs = voter_beliefs.reshape(-1, 1)
party_beliefs = np.random.normal(loc=0.0, scale=1.0, size=n_parties)
party_beliefs = party_beliefs.reshape(-1, 1)
party_popularity = np.full(n_parties, 1 / n_parties)

distances = euclidean_distances(voter_beliefs, party_beliefs)
underyling_vote = np.argmax(1 - distances, axis=1)
underlying_popularity = get_popularity(underyling_vote, n_parties=n_parties)

votes = []
popularities = []
for i_year in range(n_elections):
    vote_utilities = utility(voter_beliefs, party_beliefs, party_popularity)
    vote = np.argmax(vote_utilities, axis=1)
    votes.append(vote)
    party_popularity = get_popularity(vote, n_parties=n_parties)
    popularities.append(np.copy(party_popularity))

plot_popularities_over_time(popularities, underlying_popularity).show()
