import numpy as np
import matplotlib.pyplot as plt

#EVALUATION_KEYS = ('Danger', 'Urgency', 'Risk of Failure', 'Collaboration', 'Approachable')
def convert(values):
    trackMeans = {}
    for keyTrack, valueTrack in values:
        personTracks = [0, 0, 0, 0, 0]
        for keyPerson, valuePerson in valueTrack:
            trackData = []
            for keyData, valueData in valuePerson:
                trackData.append(valueData)
            personTracks = np.vstack(personTracks, trackData)
        averageValues = np.mean(personTracks, axis = 0)
        trackMeans['keyTrack'] = averageValues

    return trackMeans


trackMeans = convert(values)
danger = []
urgency = []
rof = []
collab = []
approach = []

for trackKey, tv in trackMeans:
    danger.append(tv[0])
    urgency.append(tv[1])
    rof.append(tv[2])
    collab.append(tv[3])
    approach.append(tv[4])

plt.boxplot([danger, urgency, rof, collab, approach])
plt.xticks([1, 2, 3, 4, 5], ['Danger', 'Urgency', 'Risk of Failure', 'Collaboration', 'Approachability'])

plt.show()
