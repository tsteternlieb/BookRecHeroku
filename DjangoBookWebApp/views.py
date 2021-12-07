from django.shortcuts import render
from DjangoBookWebApp.book_search_model import FinalWrapper
model = FinalWrapper()
# our home page view
def home(request):    
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(pclass):
    #model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    #scaled = pickle.load(open("scaler.sav", "rb"))
    #prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))
    #model = FinalWrapper()
    if pclass[0] == '"':
        pclass = pclass[1:]
    if pclass[-1] == '"':
        pclass = pclass[:-1]

    predictions = model.GetRecs(pclass)
    final = ''
    for book in predictions:
        final += book + "\n"
    return predictions
        

# our result page view
def result(request):
    pclass = str(request.GET['pclass'])

    result = getPredictions(pclass)

    return render(request, 'result.html', {'result':result})
