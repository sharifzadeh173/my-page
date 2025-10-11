from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def logisticloss(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = log_loss(y_test, y_pred)
    return loss

# Example 1
X_train1 = [[13.2], [15.2],[17.5],[19.9],[15],[19.4]]
y_train1 = [0,1,0,1,0,1]
X_test1 = [[13.2],[22.5],[14.3],[22.9]]
y_test1 = [0,0,0,1]
print("Example 1 loss:", logisticloss(X_train1, y_train1, X_test1, y_test1))

# Example 2
X_train2 = [[10.2,9],[18.2,3.3],[14,13],[19,14],[12,23],[16,17]]
y_train2 = [0,0,1,1,0,1]
X_test2 = [[10,16],[22,15],[14,23],[22,26]]
y_test2 = [0,0,0,1]
print("Example 2 loss:", logisticloss(X_train2, y_train2, X_test2, y_test2))
#output
#Example 1 loss: 9.010913347279288
#Example 2 loss: 18.021826694558577

