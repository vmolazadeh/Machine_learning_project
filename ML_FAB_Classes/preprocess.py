from sklearn.preprocessing import MinMaxScaler
class Preprocessing:
    def __init__(self, raw_data, drop=None):
        self.data = raw_data
        if drop:
            self.data = self.data.drop(columns=drop)
        
    def scaler(self):
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)
        return self.data, self.scaler
    
    def reduce(self):
        pass
    
    def do_kmeans(self):
        pass