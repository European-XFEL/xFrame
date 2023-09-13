from xframe.library.mathLibrary import masked_mean,combine_means_2D
import numpy as np
from xframe.startup_routines import dependency_injection
dependency_injection()
from xframe.database.euxfel import default_DB
from xframe.presenters.matplolibPresenter import plot1D

db =default_DB()

data = db.load('/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_32/2022-09-02_17:35:39/bg_data_proc.h5')


s = data['saxs']['saxs']
d = data['saxs']['polar_data']
m = data['saxs']['polar_mask'].astype(bool)
m[:]=d.astype(bool)
s2,counts = masked_mean(d,m,axis = 1)

fig1 = plot1D.get_fig(s)
fig2 = plot1D.get_fig(s2)


db.save('/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_32/2022-09-02_17:35:39/s1.matplotlib',fig1)
db.save('/gpfs/exfel/theory_group/user/berberic/p3046/data/fig/run_32/2022-09-02_17:35:39/s2.matplotlib',fig2)
