# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:23:02 2026

@author: dhuma
"""

s = {1,2,4,3,6,5}
print({i**2 for i in s if i%2==0 })


s={12,15,14,13,3}
print(
       {i**2 if i % 2 == 0 else i*2 for i in s}
      )