import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

cc = pd.read_csv('C:\\Users\\Jacob Mario Leonard\\Downloads\\creditcard.csv')

scaler = StandardScaler()
cc['NormalizedAmount'] = scaler.fit_transform(cc['Amount'].values.reshape(-1,1))
cc = cc.drop(['Time', 'Amount'], axis=1)

X = cc.drop('Class', axis=1)
y = cc['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

oversampler = SMOTE()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
