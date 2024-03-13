import numpy as np
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