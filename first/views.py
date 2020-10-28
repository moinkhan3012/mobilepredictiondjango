from . import price_prediction as pp
import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render,redirect
# from django_tables2.tables import Table

# Create your views here.

from io import BytesIO
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.template import RequestContext
from django.templatetags.static import static
import os
from django.contrib.staticfiles import finders
def home(request):

    knn = pp.KNN()
    lr = pp.LR()
    nbc = pp.NBC()

    plt = knn.elbow()

    buf = BytesIO()
    plt.savefig(buf,format = 'png')
    elbow = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    buf.close()

    KNNscore =knn.score

    df =pd.read_csv(finders.find('data/new_data.csv'))
    html_table = pp.showhead().to_html()

    return render(request,'home.html', {'html_table': html_table,'elbow':elbow,'KNNscore':KNNscore,'NBCscore':nbc.score,'LRscore':lr.score})

def predict_price_lr(request):
    print('redirecitng')
    return  render(request,'predict_price_lr.html')


def predict_price_nbc(request):
    print('redirecitng')
    return  render(request,'predict_price_nbc.html')



def predictlr(request):
    if request.method =="POST":
        spec = pp.Specs()
        delattr(spec,'price_range')
        spec.clock_speed = float(request.POST['CLOCK_SPEED'])
        spec.fc = float(request.POST['FRONT_CAMERA'])
        spec.int_memory = float(request.POST['INTERNAL_MEMORY'])
        spec.n_cores = float(request.POST['NUMBER_OF_CORES'])
        spec.pc = float(request.POST['PRIMARY_CAMERA'])
        spec.ram = float(request.POST['RAM']) * 1024

        lr = pp.LR()
        price = lr.LRpredict(spec)
        print(price)
    return HttpResponse(price)

def predictnbc(request):

    if request.method == "POST":
        spec = pp.Specs()
        delattr(spec, 'price_range')
        spec.clock_speed = float(request.POST['CLOCK_SPEED'])
        spec.fc = float(request.POST['FRONT_CAMERA'])
        spec.int_memory = float(request.POST['INTERNAL_MEMORY'])
        spec.n_cores = float(request.POST['NUMBER_OF_CORES'])
        spec.pc = float(request.POST['PRIMARY_CAMERA'])
        spec.ram = float(request.POST['RAM'])*1024
        nbc = pp.NBC()
        price = nbc.NBCpredict(spec)

        if  price ==1 :
            res = "0-15,000"
        elif price == 2:
            res = "15,000- 30,000"
        elif price == 3 :
            res = "30,000 - 40,000"
        else:
            res ="Greater than 40,000"

    return HttpResponse(res)

