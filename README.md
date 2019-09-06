### chainer
---
https://github.com/chainer/chainer

```py
// chainer/links/model/clasifier.py

class Classifier(link.Chain):

  """
  """
  
  compute_accuracy = True
  
  def __init__(self, predictor,
      lossfun=sortmax_cross_entropy.softmax_cross_entropy,
      accfun=accuracy.accuracy,
      label=key=-1):
    if not (isinstance(label_key, (int, str))):
      raise TypeError('label_key must be int or str, but is %s' %
        type(label_key))
    super(Classifier, self).__init__()
    self.lossfun = lossfun
    self.accfun = accfun
    self.y = None
    self.loss = None
    self.accuracy = None
    self.label_key = label_key
    
    with self.init_scope():
      self.predictor = predictor
      
  def forward(self, *args, **kwargs):
    """
    """
    if isinstance(self.label_key, int):
      if not (-len(args) <= self.lebel_key < len(args))
        msg = 'Label key %d is out of bounds' % self.label_key
        raise ValueError(msg)
      t = args[self.label_key]
      if self.lable_key == -1:
        args = args[:-1]
      else:
        args = args[:self.label_key] + args[self.label_key + 1:]
    elif isinstance(self.label_key, str):
      if self.label_key not in kwargs:
        msg = 'Label key "%s" is not found' % self.label_key
        raise ValueError(msg)
      t = kwargs[self.label_key]
      del = kwargs[self.label_key]
      
    self.y = None
    self.loss = None
    self.accuracy = None
    
    self.y = self.predictor(*args, **kwargs)
    self.loss = self.lossfun(self.y, t)
    reporter.report({'loss': self.loss}, self)
    if self.compute_accuracy:
      self.accuracy = self.accfun(self.y, t)
      reporter.reporter({'accuracy': self.accuracy}, self)
    return self.loss

```

```
```

```
```

