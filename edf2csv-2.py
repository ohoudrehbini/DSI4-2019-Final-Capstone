#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://physionet.org/content/chbmit/1.0.0/#files-panel


# In[2]:


#!pip install EEGtools


# In[39]:


# this will filter out a lot of future warnings from statsmodels
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[40]:


#!pip install pandas


# In[41]:


#! pip install numpy


# In[42]:


#! pip install matplotlib


# In[43]:


#!pip install seaborn


# In[44]:


#! pip install datetime


# In[45]:


#!pip install statsmodels


# In[46]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm


sns.set(font_scale=1.5)
plt.style.use('fivethirtyeight')

get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
get_ipython().magic(u'matplotlib inline')


sns.set(font_scale=1.5)
plt.style.use('fivethirtyeight')

get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
get_ipython().magic(u'matplotlib inline')


# In[47]:


import os 


# In[48]:


#!pip install pyEDFlib


# In[49]:


#!pip install mne


# In[50]:


#!pip install -U mne


# In[51]:


#!pip install mne


# In[52]:


#!pip install --upgrade pip


# In[53]:


#%matplotlib qt5


# In[54]:


#!pip install sklearn


# In[55]:


#!pip install scikit-learn


# In[56]:


#!pip install scipy


# In[57]:


from matplotlib import pyplot as plt
import numpy as np


# In[58]:


#!pip install base


# In[59]:


#!pip install convertfigures


# In[60]:


#!pip install data


# In[61]:


#pip._internal.utils.deprecation.deprecated. 


# In[62]:


#!pip install six


# In[63]:


#! pip install decorators


# In[64]:


get_ipython().system(u'conda info --envs')


# In[65]:


import eegtools
import pandas as pd

class EDF:

    def __init__(self, subject):

        self.subject = subject

        edf_data = eegtools.io.load_edf(self.subject)

        self.data = edf_data.X.transpose()
        self.smp_rate = edf_data.sample_rate
        self.channels = edf_data.chan_lab
        self.ann = edf_data.annotations

    def signal_to_csv(self):
        '''
          To save the dataframe
        '''
        df = pd.DataFrame(data = self.data, columns = self.channels)
        df.to_csv('%s.csv' % self.subject, index = False, encoding='utf-8')

    def ann_to_csv(self):
        '''
          To save the annotations
        '''
        ann = pd.DataFrame(data = self.ann, columns = ['time', 'duration', 'label'])

        for i in range(len(ann)):
          ann.loc[i, 'label'] = ann.loc[i, 'label'][0]

        ann.to_csv('%s_ann.csv' % self.subject, index = False, encoding='utf-8')

    def signal(self):
        return self.data

    def info(self):
        return [self.smp_rate, self.channels, self.ann]


# In[ ]:





# In[ ]:




