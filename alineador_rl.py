# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:26:58 2020

@author: angel
"""
import time
import numpy as np
import progressbar
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow import keras
import cppyy
cppyy.cppdef("""
using namespace std;
vector<double> NW(vector<double> S, vector<double> p1, vector<double> p2) {
  int i,j,k,r;
  int aux=S.size();
  int Q;
  if(aux==25) Q=5;
  else Q=21;
  int l1= p1.size()/Q;
  int l2=p2.size()/Q;
  vector<double> ans( (l1+1)*(l2+1) ), prev( (l1+1)*(l2+1) );
  ans[0]=0;
  prev[0]=-1;
  for(j=1;j<=l2;j++) {
    ans[(l2+1)*(0)+(j)]=ans[(l2+1)*(0)+(j-1)];
    prev[(l2+1)*(0)+(j)]=0;
    for(k=0;k<Q;k++){
      ans[(l2+1)*(0)+(j)]+=S[(Q)*(k)+(Q-1)]*p2[(Q)*(j-1)+(k)];
    }  
  }
  for(i=1;i<=l1;i++) {
    ans[(l2+1)*(i)+(0)]=ans[(l2+1)*(i-1)+(0)];
    prev[(l2+1)*(i)+(0)]=2;
    for(k=0;k<Q;k++){
      ans[(l2+1)*(i)+(0)]+=S[(Q)*(k)+(Q-1)]*p1[(Q)*(i-1)+(k)];
    }  
  }
  double aux1,aux2;
  for(i=1;i<=l1;i++) {
    for(j=1;j<=l2;j++) {
      ans[(l2+1)*(i)+(j)]=ans[(l2+1)*(i-1)+(j-1)];
      prev[(l2+1)*(i)+(j)]=1;
      for(k=0;k<Q;k++) {
        for(r=0;r<Q;r++) {
          ans[(l2+1)*(i)+(j)]+=p1[(Q)*(i-1)+(k)]*S[(Q)*(k)+(r)]*p2[(Q)*(j-1)+(r)]; 
        }
      }
      aux1=ans[(l2+1)*(i)+(j-1)];
      for(k=0;k<Q;k++) {
        aux1+=S[(Q)*(k)+(Q-1)]*p2[(Q)*(j-1)+(k)];
      }
      aux2=ans[(l2+1)*(i-1)+(j)];
      for(k=0;k<Q;k++) {
        aux2+=S[(Q)*(k)+(Q-1)]*p1[(Q)*(i-1)+(k)];
      }
      if(aux1>ans[(l2+1)*(i)+(j)]) {
        ans[(l2+1)*(i)+(j)]=aux1;
        prev[(l2+1)*(i)+(j)]=0;
      }
      if(aux2>ans[(l2+1)*(i)+(j)]) {
        ans[(l2+1)*(i)+(j)]=aux2;
        prev[(l2+1)*(i)+(j)]=2; 
      } 
    }
  }
  vector<double> cam;
  i=l1;
  j=l2;
  while( i!=0 || j!=0) {
    cam.push_back(prev[(l2+1)*(i)+(j)]);
    if(prev[(l2+1)*(i)+(j)]==1) {
      i--;
      j--;
    } 
    else {
      if(prev[(l2+1)*(i)+(j)]==0) {
        j--;
      }
      else {
        i--;
      }
    } 
  }
  return cam;
}
""")
class AlineadorRL:
  def __init__(self,sequenceType='DNA',matchScore=2, unMatchScore=-1, matchGapScore=-2,RLmethod='Tree'):
    self.sequenceType = sequenceType
    self.matchScore = matchScore
    self.unMatchScore = unMatchScore
    self.matchGapScore=matchGapScore
    self.RLmethod = RLmethod
    if self.sequenceType == 'DNA':
      self.Q = ['G','A','T','C','-']
    if self.sequenceType == 'RNA':
      self.Q = ['G','A','U','C','-']
    if self.sequenceType == 'AMINOACID':
      self.Q = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-']
    self.mapQ ={}
    for i in range(len(self.Q)): self.mapQ[self.Q[i]]=i
    Qlen = len(self.Q)-1
    S= np.zeros( (Qlen+1,Qlen+1) )
    for i in range(Qlen+1):
      for j in range(Qlen+1):
        if (i== Qlen and j != Qlen ) or (i!= Qlen and j == Qlen) :
          S[i,j]= self.matchGapScore
          continue
        if i == j and i == Qlen :
          S[i,j] = 0
          continue
        if i == j :
          S[i,j] = self.matchScore
        else :
          S[i,j] = self.unMatchScore
    self.S = S

  

  def scoreProfileVectors(self, v1,v2):
    ans = np.matmul(v1,self.S)
    ans = np.matmul(ans,v2)
    return ans
  
  def scoreProfileVectorGap(self,v):
    ans = self.S[len(self.S)-1].dot(v)
    return ans

  def blockToProfile(self,B):
    cant = len(B)
    per = np.zeros( (len(self.Q), len(B[0]) ) )
    for i in range(len(B)):
      for j in range(len(B[i])):
        per[ self.mapQ[B[i][j]] ,j]+=1
    per = per / cant
    return per.T

  def SPscore(self,B,r=-1):
    ans = 0
    for k in range(len(B[0])):
      for i in range(len(B)-1):
        for j in range(i+1,len(B)):
          if r== -1: 
            ans+=self.S[self.mapQ[B[i][k]], self.mapQ[B[j][k]]]
          else:
            if i<= len(B)-r-1 and j >= len(B)-r:
              ans+=self.S[self.mapQ[B[i][k]], self.mapQ[B[j][k]]]
    return ans

  def metricas(self,B):
    EM=0
    for i in range(len(B[0])):
      localEM=1
      if len(B) > 1:
        for j in range(1,len(B)):
          if B[j][i]!=B[0][i]: localEM=0
      EM+=localEM
    AL = len(B[0])
    CS = EM / AL
    return self.SPscore(B), AL , EM, CS

  def SPvector(self,v):
    ans = 0
    for i in range(len(v)-1):
      for j in range(i+1,len(v)):
        ans+=self.S[self.mapQ[v[i]], self.mapQ[v[j]]]
    return ans

  def alignProfileNW(self,B1,B2):
    # S matriz de score Q + 1 x Q + 1
    # B1 B2 bloques previamente alineados
    p1=self.blockToProfile(B1)
    p2=self.blockToProfile(B2)
    l1 = p1.shape[0]
    l2 = p2.shape[0]
    gap = np.zeros_like(p1[0])
    gap[len(self.Q)-1]=1
    # NW para guardar los valores de la DP
    NW = np.zeros( (l1+1,l2+1) )
    # NWprev para guardar la opcion que llevo a un maximo.
    # 0 'izquierda' , 1 'diagonal', 2 'arriba'
    NWprev = np.zeros_like(NW)
    ## Casos Base, prejifos vacios
    NW[0,0]= 0
    NWprev[0,0] = -1
    ## Casos Base, Prefijo no vacio contra vacio. Rellenar con gaps
    for i in range(1,l1+1):
      NW[i,0]=NW[i-1,0]+self.scoreProfileVectorGap(p1[i-1])
      NWprev[i,0]=2
    for j in range(1,l2+1):
      NW[0,j]=NW[0,j-1]+self.scoreProfileVectorGap(p2[j-1])
      NWprev[0,j] = 1
    ## Caso recursivo
    for i in range(1,l1+1):
      for j in range(1,l2+1):
        NW[i,j] = NW[i-1,j-1]+self.scoreProfileVectors(p1[i-1],p2[j-1])
        NWprev[i,j] = 1
        aux2 = NW[i-1,j] + self.scoreProfileVectorGap(p1[i-1])
        aux0 = NW[i,j-1] + self.scoreProfileVectorGap(p2[j-1])
        if aux2 > NW[i,j]:
          NW[i,j]= aux2
          NWprev[i,j] = 2
        if aux0 > NW[i,j]:
          NW[i,j]= aux0
          NWprev[i,j] = 0
    ## Reconstruccion del camino
    fil=l1
    col = l2
    camino = []
    while NWprev[fil,col]!= -1:
      camino.append(NWprev[fil,col])
      if NWprev[fil,col] == 0:
        col-=1
      else:
        if NWprev[fil,col] == 1:
          fil-=1
          col-=1
        else:
          fil-=1
    camino = np.flip(np.array(camino))
    rows = len(B1)+len(B2)
    B=[]
    for i in range(rows):
      B.append("")
    i=0
    j=0
    for mov in camino:
      for x in range(len(B1)):
        if mov == 2 or mov == 1:
          B[x]+=B1[x][i]
        else:
          B[x]+='-'
      for x in range(len(B2)):
        if mov == 0 or mov == 1:
          B[len(B1)+x]+=B2[x][j]
        else:
          B[len(B1)+x]+='-'
      if mov == 2 or mov == 1:
        i+=1
      if mov == 0 or mov == 1:
        j+=1
    return B

  def alignProfileNWcpp(self,B1,B2):
    # S matriz de score Q + 1 x Q + 1
    # B1 B2 bloques previamente alineados
    p1=self.blockToProfile(B1)
    p2=self.blockToProfile(B2)
    camcpp = cppyy.gbl.NW(self.S.flatten(),p1.flatten(),p2.flatten())
    camino=np.flip(np.array(list(camcpp),dtype=np.int))
    rows = len(B1)+len(B2)
    B=[]
    for i in range(rows):
      B.append("")
    i=0
    j=0
    for mov in camino:
      for x in range(len(B1)):
        if mov == 2 or mov == 1:
          B[x]+=B1[x][i]
        else:
          B[x]+='-'
      for x in range(len(B2)):
        if mov == 0 or mov == 1:
          B[len(B1)+x]+=B2[x][j]
        else:
          B[len(B1)+x]+='-'
      if mov == 2 or mov == 1:
        i+=1
      if mov == 0 or mov == 1:
        j+=1
    return B

  def create_modelPermutation(self,cant_sec):
    # Inputs
    inputAction = keras.Input(shape=(cant_sec,),name = 'Action')
    inputState = keras.Input(shape = (cant_sec,), name = 'State') 
    # Proceso Action
    outAction = layers.Dense(16,activation="relu")(inputAction)
    # Proceso State
    emb1= layers.Embedding(input_dim=cant_sec+1, output_dim=5) (inputState)
    outState = layers.LSTM(16) (emb1)
    # Combinar
    comb = layers.concatenate([outAction,outState])
    combiD1= layers.Dense(16,activation="relu") (comb)
    outQ = layers.Dense(1) (combiD1)
    Q = keras.Model(inputs = [inputAction,inputState], outputs = [outQ])
    Q.compile(optimizer='adam',loss = 'mean_squared_error')
    return Q  
  def getQvalsPermutation(self,Qfun,actualState,validAction,maskAction) :
    qState = np.array([ actualState for x in validAction])
    qAction = np.array([maskAction[x] for x in validAction])
    qvals = np.array([x[0] for x in Qfun({'State':qState,'Action':qAction})])
    return qvals
  def buildAlignmentPermutation(self,sec,Qfun,maskAction):
    N = len(sec)
    actualState = np.zeros((N))
    pos = 0
    while 0 in actualState:
      validAction = np.array(list(filter( lambda y: y+1 not in actualState , 
                                           [x for x in range(len(maskAction))])))
      qvals = self.getQvalsPermutation(Qfun,actualState,validAction,maskAction) 
      a = int(np.argmax(qvals))
      a = validAction[a]
      actualState[pos] = a+1
      if pos == 0:
        B = [ sec[a] ]
      else :
        B = self.alignProfileNWcpp(B,[sec[a]])
      pos = pos + 1
    return B
  def alignPermutation(self,sec,sizeExp=100, sizeMiniBatch = 100, M= 20, epsilon = 0.2, 
                       gamma = 0.1,alpha=0.1,patience=15,check_convergence=False,epocas=100, returnbestCS = False):
    seed(42)
    set_seed(42)
    iniTiempo=time.time()
    # Precalculos del enfoque
    N = len(sec)
    Qfun = self.create_modelPermutation(N)
    maskAction = np.array([ [  x if x==i  else 0 for x in range(N)] for i in range(N)])
    metricasGlobal = []
    firstMetrica = True
    # Precalculos convergencia
    lastCheck = -1
    bestScore = 0
    bestCS = 0
    conv=False
    trained = False
    # Inicializar repositorio experiencias
    currentSizeExp=0
    nextExpIndex=0
    experience = np.zeros((sizeExp,N+2)) # N state, 1 action, 1 reward
    ep = 0
    # Inicia Q - Learning
    bar = progressbar.ProgressBar(maxval = M,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    while ep<M and conv == False:
      # Inicia Episodio
      # Precalculos Episodio
      actualState = np.zeros((N)) # estado actual, prefijo de permutacion del 1 al N. 
                                  # Se rellena el sufijo con ceros
      pos = 0 # Proximo indice a considerar
      while 0 in actualState:
        # Inicia Transicion
        # Inicio criterio de exploracion
        validAction = np.array(list(filter( lambda y: y+1 not in actualState , 
                                           [x for x in range(len(maskAction))])))
        criterio = np.random.uniform(0,1)
        if criterio <= epsilon:
          a = np.random.randint(0,len(validAction))
        else:
          qvals = self.getQvalsPermutation(Qfun,actualState,validAction,maskAction) 
          a = int(np.argmax(qvals))
        a = validAction[a] # entre 0 y len(maskAction)
        # Fin criterio de exploracion
        # Inicia ejecucion de accion
        exp = actualState
        exp= np.append(exp,a)
        actualState[pos] = a+1
        if pos == 0:
          B = [ sec[a] ]
          exp =np.append(exp,0)
        else :
          B = self.alignProfileNWcpp(B,[sec[a]])
          exp = np.append(exp,self.SPscore(B,1))
        experience[nextExpIndex] = exp
        nextExpIndex = nextExpIndex + 1
        if nextExpIndex == sizeExp: nextExpIndex = 0
        currentSizeExp = currentSizeExp + 1
        pos = pos + 1
        # Fin ejecucion de accion
        # Inicia experience replay
        if currentSizeExp >= sizeMiniBatch:
          indExp = np.random.randint(0,min(currentSizeExp,sizeExp),sizeMiniBatch)
          qState = []
          qAction = []
          qTarget = []
          # Inicia Obtencion de bigQvals
          tamBatch = np.zeros((len(indExp)),dtype = np.int32)
          qBatchState = []
          qBatchAction = []
          for i,j in zip(indExp,range(len(indExp))):
            st = np.array(experience[i][0:N])
            act = int(experience[i][N])
            stD = np.array(st)
            w =0 
            while stD[w] != 0: 
              w = w+1
            stD[w]=act+1
            if 0 not in stD:
              qBatchState.append(st)
              qBatchAction.append(maskAction[act])
              tamBatch[j] = 0
            else:
              validActionD = np.array(list(filter( lambda y: y+1 not in stD , 
                                           [x for x in range(len(maskAction))])))
              qBatchState.append(st)
              qBatchAction.append(maskAction[act])
              for w in validActionD:
                qBatchState.append(stD)
                qBatchAction.append(maskAction[w])
              tamBatch[j] = len(validActionD)
          qBatchState = np.array(qBatchState)
          qBatchAction = np.array(qBatchAction)
          bigQvals = np.array([x[0] for x in Qfun({'State':qBatchState,'Action':qBatchAction})])
          iniBigQvals=0
          # Fin obtencion de bigQvals
          for i,j in zip(indExp,range(len(indExp))):
            st = np.array(experience[i][0:N])
            act = int(experience[i][N])
            rew = experience[i][N+1]
            qval = bigQvals[iniBigQvals]
            iniBigQvals= iniBigQvals+1
            stD = np.array(st)
            w =0 
            while stD[w] != 0: 
              w = w+1
            stD[w]=act+1
            if 0 not in stD: # stD estado Final
              targ = rew
            else:
              qvalsD = bigQvals[iniBigQvals:(iniBigQvals+tamBatch[j])]
              iniBigQvals = iniBigQvals + tamBatch[j] 
              targ = rew + gamma*np.max(qvalsD)
            targ = alpha*targ + (1-alpha)*qval
            qState.append(st)
            qAction.append(maskAction[act])
            qTarget.append(targ)
          # Inicio fit
          qState=np.array(qState)
          qAction =np.array(qAction)
          qTarget=np.array(qTarget)
          Qfun.fit({'State':qState,'Action':qAction},y=qTarget, verbose=0,epochs = epocas)
          trained=True
          # Fin fit
        # Termina experience replay
        # Termina Transicion 
      ep = ep + 1
      B = self.buildAlignmentPermutation(sec,Qfun,maskAction)
      metricasEpisodio = self.metricas(B) 
      if trained == True : 
        metricasGlobal.append(metricasEpisodio)
      else:
        if firstMetrica == True:
          metricasGlobal.append(metricasEpisodio)
          firstMetrica = False
          if lastCheck == -1:
            lastCheck = ep
            bestScore = metricasEpisodio[0]
            bestCS = metricasEpisodio[3]
            Qfun.save("auxiliar_model.h5")
      # Termina Episodio
      # Inicio Revisar convergencia
      if check_convergence==True and trained==True:
        if lastCheck == -1:
          lastCheck = ep
          bestScore = metricasEpisodio[0]
          bestCS = metricasEpisodio[3]
          Qfun.save("auxiliar_model.h5")
        else :
          localScore = metricasEpisodio[0]
          localCS = metricasEpisodio[3]
          if returnbestCS == False:
            if localScore< bestScore or ( localScore == bestScore and localCS <=  bestCS) :
              if ep - lastCheck > patience:
                conv= True
            else :
              bestScore = localScore
              bestCS = localCS
              lastCheck = ep
              Qfun.save("auxiliar_model.h5")
          else:
            if localCS< bestCS or ( localCS == bestCS and localScore <=  bestScore) :
              if ep - lastCheck > patience:
                conv= True
            else :
              bestScore = localScore
              bestCS = localCS
              lastCheck = ep
              Qfun.save("auxiliar_model.h5")
      # Fin Revisar convergencia
      bar.update(ep)
    # Termina Q - Learning
    if check_convergence == True:
      Qfun = load_model("auxiliar_model.h5")
    B = self.buildAlignmentPermutation(sec,Qfun,maskAction)
    print( self.metricas(B))
    print("Tiempo ", time.time()-iniTiempo, " segundos")
    bar.finish()
    return B , Qfun , np.array(metricasGlobal) 
  def create_modelTree(self,cant_sec):
    # Inputs
    inputAction = keras.Input(shape=(cant_sec,),name = 'Action')
    inputState = keras.Input(shape = (cant_sec,), name = 'State') 
    # Proceso Action
    outAction = layers.Dense(16,activation="relu")(inputAction)
    # Proceso State
    emb1= layers.Embedding(input_dim=cant_sec, output_dim=5) (inputState)
    outState = layers.LSTM(16) (emb1)
    # Combinar
    comb = layers.concatenate([outAction,outState])
    combiD1= layers.Dense(16,activation="relu") (comb)
    outQ = layers.Dense(1) (combiD1)
    Q = keras.Model(inputs = [inputAction,inputState], outputs = [outQ])
    Q.compile(optimizer='adam',loss = 'mean_squared_error')
    return Q
    
  def getQvalsTree(self,Qfun,actualState,validAction,maskAction) :
    qState = np.array([ actualState for x in validAction])
    qAction = np.array([ np.array([ 0 if i !=maskAction[x,0] and i!=maskAction[x,1] else 1 for i in range(len(actualState))])  
                        for x in validAction])
    qvals = np.array([x[0] for x in Qfun({'State':qState,'Action':qAction})])
    return qvals
  def buildAlignmentTree(self,sec,Qfun,maskAction):
    N = len(sec)
    actualState = np.array([x for x in range(N)],dtype = np.int32) 
    meta = np.array([0 for x in range(len(sec) )],dtype=np.int32)
    perfiles = [ [x] for x in sec ] 
    while sum(meta!=actualState)!=0:
      validAction = np.array(list(filter( lambda y: maskAction[y,0] in actualState and maskAction[y,1] in actualState, 
                                           [x for x in range(len(maskAction))])))
      qvals = self.getQvalsTree(Qfun,actualState,validAction,maskAction) 
      a = int(np.argmax(qvals))
      a = validAction[a]
      perfiles[maskAction[a,0]] = self.alignProfileNWcpp(perfiles[maskAction[a,0]],perfiles[maskAction[a,1]])
      actualState[actualState == maskAction[a,1] ] = maskAction[a,0]
    return perfiles[0]
  def alignTree(self,sec,sizeExp=100, sizeMiniBatch = 100, M= 20, epsilon = 0.2, 
                       gamma = 0.1,alpha=0.1,patience=15,check_convergence=False,epocas=100,returnbestCS = False):
    seed(42)
    set_seed(42)
    iniTiempo=time.time()
    # Precalculos del enfoque
    N = len(sec)
    numEstados = int((N*(N-1)) /2)
    Qfun = self.create_modelTree(N)
    maskAction= np.zeros((numEstados,2),dtype=np.int32)
    i=0
    for x in range(N):
      for y in range(N):
        if x<y:
          maskAction[i,0]=x
          maskAction[i,1]=y
          i+=1
    meta = np.array([0 for x in range(len(sec) )],dtype=np.int32)
    metricasGlobal = []
    firstMetrica = True
    # Precalculos convergencia
    lastCheck = -1
    bestScore = 0
    conv=False
    bestCS = 0
    trained = False
    # Inicializar repositorio experiencias
    currentSizeExp=0
    nextExpIndex=0
    experience = np.zeros((sizeExp,N+2)) # N state, 1 action, 1 reward
    ep = 0
    # Inicia Q - Learning
    bar = progressbar.ProgressBar(maxval = M,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    while ep<M and conv == False:
      # Inicia Episodio
      # Precalculos Episodio
      actualState = np.array([x for x in range(N)],dtype = np.int32) 
      perfiles = [ [x] for x in sec ] 
      while sum(meta!=actualState)!=0:
        # Inicia Transicion
        # Inicio criterio de exploracion
        validAction = np.array(list(filter( lambda y: maskAction[y,0] in actualState and maskAction[y,1] in actualState, 
                                           [x for x in range(len(maskAction))])))
        criterio = np.random.uniform(0,1)
        if criterio <= epsilon:
          a = np.random.randint(0,len(validAction))
        else:
          qvals = self.getQvalsTree(Qfun,actualState,validAction,maskAction) 
          a = int(np.argmax(qvals))
        a = validAction[a] 
        # Fin criterio de exploracion
        # Inicia ejecucion de accion
        perfiles[maskAction[a,0]] = self.alignProfileNWcpp(perfiles[maskAction[a,0]],perfiles[maskAction[a,1]])
        exp = actualState
        exp= np.append(exp,a)
        actualState[actualState == maskAction[a,1] ] = maskAction[a,0]
        exp = np.append(exp,self.SPscore(perfiles[maskAction[a,0]],len(perfiles[maskAction[a,1]]) ) )
        experience[nextExpIndex] = exp
        nextExpIndex = nextExpIndex + 1
        if nextExpIndex == sizeExp: nextExpIndex = 0
        currentSizeExp = currentSizeExp + 1
        # Fin ejecucion de accion
        # Inicia experience replay
        if currentSizeExp >= sizeMiniBatch:
          indExp = np.random.randint(0,min(currentSizeExp,sizeExp),sizeMiniBatch)
          qState = []
          qAction = []
          qTarget = []
          # Inicia Obtencion de bigQvals
          tamBatch = np.zeros((len(indExp)),dtype = np.int32)
          qBatchState = []
          qBatchAction = []
          for i,j in zip(indExp,range(len(indExp))):
            st = np.array(experience[i][0:N])
            act = int(experience[i][N])
            stD = np.array(st)
            stD [stD == maskAction[act,1]] = maskAction[act,0] 
            if sum(meta!=stD)==0:
              qBatchState.append(st)
              qBatchAction.append(np.array([ 0 if y !=maskAction[act,0] and y!=maskAction[act,1] else 1 for y in range(N)] ,dtype=np.int32))
              tamBatch[j] = 0
            else:
              validActionD = np.array(list(filter( lambda y: maskAction[y,0] in stD and maskAction[y,1] in stD, 
                                           [x for x in range(len(maskAction))])))
              qBatchState.append(st)
              qBatchAction.append(np.array([ 0 if y !=maskAction[act,0] and y!=maskAction[act,1] else 1 for y in range(N)] ,dtype=np.int32))
              for w in validActionD:
                qBatchState.append(stD)
                qBatchAction.append(np.array([ 0 if y !=maskAction[w,0] and y!=maskAction[w,1] else 1 for y in range(N)] ,dtype=np.int32))
              tamBatch[j] = len(validActionD)
          qBatchState = np.array(qBatchState)
          qBatchAction = np.array(qBatchAction)
          bigQvals = np.array([x[0] for x in Qfun({'State':qBatchState,'Action':qBatchAction})])
          iniBigQvals=0
          # Fin obtencion de bigQvals
          for i,j in zip(indExp,range(len(indExp))):
            st = np.array(experience[i][0:N])
            act = int(experience[i][N])
            rew = experience[i][N+1]
            qval = bigQvals[iniBigQvals]
            iniBigQvals= iniBigQvals+1
            stD = np.array(st)
            stD [stD == maskAction[act,1]] = maskAction[act,0] 
            if sum(meta!=stD)==0: # stD estado Final
              targ = rew
            else:
              #validActionD = np.array(list(filter( lambda y: maskAction[y,0] in stD and maskAction[y,1] in stD, 
              #                             [x for x in range(len(maskAction))])))
              qvalsD = bigQvals[iniBigQvals:(iniBigQvals+tamBatch[j])]
              iniBigQvals = iniBigQvals + tamBatch[j]
              targ = rew + gamma*np.max(qvalsD)
            targ = alpha*targ + (1-alpha)*qval
            qState.append(st)
            qAction.append(np.array([ 0 if y !=maskAction[act,0] and y!=maskAction[act,1] else 1 for y in range(N)] ,dtype=np.int32) )
            qTarget.append(targ)
          # Inicio fit
          qState=np.array(qState)
          qAction =np.array(qAction)
          qTarget=np.array(qTarget)
          Qfun.fit({'State':qState,'Action':qAction},y=qTarget, verbose=0,epochs = epocas)
          trained = True
          # Fin fit
        # Termina experience replay
        # Termina Transicion 
      ep = ep + 1
      B = self.buildAlignmentTree(sec,Qfun,maskAction)
      metricasEpisodio = self.metricas(B)
      if trained == True : 
        metricasGlobal.append(metricasEpisodio)
      else:
        if firstMetrica == True:
          metricasGlobal.append(metricasEpisodio)
          firstMetrica = False
          if check_convergence==True and lastCheck == -1:
            lastCheck = ep
            bestScore = metricasEpisodio[0]
            bestCS = metricasEpisodio[3]
            Qfun.save("auxiliar_model.h5") 
      # Termina Episodio
      # Inicio Revisar convergencia
      if check_convergence==True and trained==True:
        if lastCheck == -1:
          lastCheck = ep
          bestScore = metricasEpisodio[0]
          bestCS = metricasEpisodio[3]
          Qfun.save("auxiliar_model.h5")
        else :
          localScore = metricasEpisodio[0]
          localCS = metricasEpisodio[3]
          if returnbestCS == False:
            if localScore< bestScore or ( localScore == bestScore and localCS <=  bestCS) :
              if ep - lastCheck > patience:
                conv= True
            else :
              bestScore = localScore
              bestCS = localCS
              lastCheck = ep
              Qfun.save("auxiliar_model.h5")
          else:
            if localCS< bestCS or ( localCS == bestCS and localScore <=  bestScore) :
              if ep - lastCheck > patience:
                conv= True
            else :
              bestScore = localScore
              bestCS = localCS
              lastCheck = ep
              Qfun.save("auxiliar_model.h5")
      # Fin Revisar convergencia
      bar.update(ep)
    # Termina Q - Learning
    if check_convergence == True:
      Qfun = load_model("auxiliar_model.h5")
    B = self.buildAlignmentTree(sec,Qfun,maskAction)
    print( self.metricas(B))
    print("Tiempo ", time.time()-iniTiempo, " segundos")
    bar.finish()
    return B , Qfun , np.array(metricasGlobal)

  def buildAlignment(self,Q,sec):
    if self.RLmethod =='Tree':
      return self.buildAlignmentTree(Q,sec)
    if self.RLmethod =='Permutation':
      return self.buildAlignmentPermutation(Q,sec)
    if self.RLmethod == 'Suffix':
      return self.buildAlignmentSuffixV2(Q,sec)
    
  def align(self, sec, sizeExp=100, sizeMiniBatch = 100, M= 20, epsilon = 0.2, gamma = 0.99,alpha=0.1,patience=15,check_convergence=True):
    if self.RLmethod == 'Tree':    
      return self.alignTree(sec, sizeExp, sizeMiniBatch, M, epsilon, gamma,alpha,patience,check_convergence)
    if self.RLmethod == 'Permutation':    
      return self.alignPermutation(sec, sizeExp, sizeMiniBatch, M, epsilon, gamma,alpha,patience,check_convergence)
    if self.RLmethod == 'Suffix':    
      return self.alignSuffixV2(sec, sizeExp, sizeMiniBatch, M, epsilon, gamma,alpha,patience,check_convergence)

