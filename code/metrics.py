import numpy as np

class performanceMetric():
    """
        To collect performance metrices, including:
        ext_infos (sample id) is for computing overlapping rate,
        won't compute if don't provide this parameter.
        Usage:
            m_a = performanceMetric()
            
            m_a.update(preidctions, labesl, ext_infos)
    """
    def __init__(self, n_classes, name = None):
        if name is not None:
            self.name = name
        else:
            self.name = 'm_default'
            
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.num = [0] * n_classes
        self.acc = [0] * n_classes
        # the average predicting probability of class i is classified as class
        self.prob = np.zeros((n_classes, n_classes))
        # for compute overlapping rate
        self.class_wise_sample_dict = [{} for _ in range(n_classes)]

        # loss
        self.loss = 0
    
    def update(self, predictions, labels, ext_infos = None, loss = None):
        """
            ext_infos is used for computing 
            overlapping rate between two modality
        """
        if loss is not None:
            self.loss += loss.item()
        # optimize (1): to save time transfering data from gpu to cp
        predictions = predictions.cpu().data.numpy()
        idxs = np.argmax(predictions, axis=1)
        labels = labels.cpu().data.numpy()
        
        for i in range(len(labels)):
            label = labels[i]
            prediction = predictions[i]
            self.num[label] += 1
            
            if ext_infos is not None:
                sample_id = ext_infos[i]
                if  sample_id not in self.class_wise_sample_dict[label]:
                    self.class_wise_sample_dict[label][sample_id] = 0
                else:
                    # print('Duplicate sample id:', sample_id)
                    pass
            # optimize (1)
            # idx = np.argmax(prediction.cpu().data.numpy())
            # self.prob[label][idx] += prediction.cpu().data.numpy()[idx]
            idx = idxs[i]
            self.prob[label][idx] += prediction[idx]

            if label == idx:
                self.acc[label] += 1
                if ext_infos is not None:
                    self.class_wise_sample_dict[label][sample_id]= 1
            self.confusion_matrix[label][idx] += 1

    def get_acc(self):
        return round(np.sum(self.acc) / np.sum(self.num), 4)
    
    def get_class_acc(self):
        return [round(i,4) for i in np.array(self.acc) / np.array(self.num)]

    def compute_prob(self):
        """
            When you want to compute the average probability
            of preidction for each class. 
            Use it after update()
        """

        prob = np.zeros((self.n_classes,self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if self.confusion_matrix[i,j] == 0:
                    prob[i,j] = 0
                else:
                    prob[i,j] = round(self.prob[i,j] / self.confusion_matrix[i,j], 4)
        return prob

    def comput_class_avg_prob(self):
        # eg. if the model is 100% sure about the prediction, the probabily is 1
        # if the model is 50% sure about the prediction, the probabily is 0.5
        # the average probabily is the average of the probabily of the correct prediction
        prob_matrics = self.compute_prob()
        class_prob = [prob_matrics[i][i] for i in range(len(prob_matrics))]
        avg_prob = np.mean(class_prob)
        return round(avg_prob, 4)