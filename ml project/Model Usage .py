#!/usr/bin/env python
# coding: utf-8

# ## Model Testing

# In[1]:


from joblib import dump, load
model=load("RealEstate.joblib")


# In[5]:


import numpy as np
features=np.array([[-0.54230647,  7.18716752, -1.12669699, -0.27288841, -1.42019852,
       -0.74385314, -1.7372291 ,  2.56306103, -0.99293673, -0.57167103,
       -0.99480936,  0.4386066 , -0.49636234]])
model.predict(features)


# In[ ]:





# In[ ]:




