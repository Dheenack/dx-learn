from dxlearn import DXClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

X,y=load_iris(return_X_y=True)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=7,test_size=0.3)

model= DXClassifier(population_size=20,generations=20,cv=5,alpha=1.0,beta=0.2,\
gamma=0.01,max_runtime=600,verbose=0,n_jobs=-1,deterministic=True,random_state=7)

model.fit(xtrain,ytrain)

preds=model.predict(xtest)

print(f"model best score={model.best_score_}")
print(model.score(xtest, ytest))
print(model.best_pipeline_)
# Optional: model.dashboard()  # starts blocking server; run manually if needed
