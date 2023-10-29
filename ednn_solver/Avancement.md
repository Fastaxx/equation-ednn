## Avancement

- Bibliographie
- Implémentation : 
	- Décomposition de l'équation de GL en partie réel / imaginaire
	- Réseau de neurones : 2 outputs : 1 partie Réel / 1 partie Imaginaire
	- Evaluation du RHS : Hardcoded 2 equations
- Test

### Compréhension

On crée le modèle : Réseau de neurones : Feed Forward Neural Network (FFNN)
```
# Input definition
        coords = keras.layers.Input(self.din, name='coords')

        # Normalzation
        hidden  = coords

        # Hidden layers
        for ii in range(depth):
            hidden = keras.layers.Dense(width)(hidden)
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hidden = self.act_fn(hidden)

        # Output definition
        fields = keras.layers.Dense(self.dout, name='fields')(hidden)

        cte   = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(coords)
        dummy = keras.layers.Dense(1, use_bias=False)(cte)
        self.inv_outputs = [dummy]

        # Create model
        model = keras.Model(inputs=coords, outputs=[fields]+self.inv_outputs)
        self.model = model
```

On découpe le jeu d'entrainement en mini batches :
```
# Create batches and cast to TF objects
                (X_batch,
                 Y_batch) = get_mini_batch(X_data,
                                          Y_data,
                                          ba,
                                          batches,
                                          flag_idxs,
                                          random=rnd_order_training)

                X_batch = tf.convert_to_tensor(X_batch)
                Y_batch = tf.convert_to_tensor(Y_batch)
                ba_counter = tf.constant(ba)
```

On entraine le modèle ('ajustements des paramètres du réseaux en minimisant la fonction cout'):
```
 loss_data = self.training_step_gl(X_batch, Y_batch)
                print ('ep :',ep,'ba :', ba,'loss_data :', loss_data.numpy())
```
On évalue les sorties prédites puis on les compares avec le jeu d'entrainement. On calcule la MSE. On récupére les gradients.
```
# For training of the EDNN at initial time on a batch of data. 
    @tf.function
    def training_step_gl(self, X_batch, Y_batch):
        with tf.GradientTape(persistent=True) as tape:
            Ypred_real = self.output(X_batch)[0]
            Ypred_img = self.output(X_batch)[1]
            aux_real = [tf.reduce_mean(tf.square(Ypred_real[i] - Y_batch[i,:])) for i in range(len(Ypred_real))]
            aux_img = [tf.reduce_mean(tf.square(Ypred_img[i] - Y_batch[i,:])) for i in range(len(Ypred_img))]
            aux = aux_real+aux_img
            loss_data = tf.add_n(aux)
            loss = loss_data
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape
        gradients = [x for x in gradients_data]
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        return loss_data
```

Une fois le modèle entrainée (weights et biais ajustés) On fait marcher ('Marching') le réseau dans le temps.

```
# March the EDNN class till Nt time steps. 
        for n in range(nrestart+1,Nt):
            print('time step', n)
            EDNN.Marching(Input,params_marching)

            # The real and imaginary field is stored every time step.
            [U,V] = EDNN.output(Input)
            U = U.numpy().reshape(Nx)
            V = V.numpy().reshape(Nx)
```

```
# Higher level function to execute the time marching.         
    def Marching(self, Input, params):
        W = self.get_weights_np()
        W = self.marching_method(W,self.eval_rhs,Input,params)
        self.set_weights_np(W)
        return None
```

On fait marcher le réseau selon un schéma suivant : 
```
# -----------------------------------------------------------------------------
# Time marching schemes 
# -----------------------------------------------------------------------------
def Forward_Euler(w, eval_rhs, Input,params):
    dt = params[0]
    nbatch = params[1]
    dwdt = eval_rhs(Input,nbatch)
    
    dw = dt * dwdt
    w += dw 
    return w

def Runge_Kutta(w,eval_rhs,Input,params):
    dt = params[0]
    nbatch = params[1]
    c = [1.0/8.0,3.0/8.0,3.0/8.0,1.0/8.0]

    k1 = eval_rhs(Input,nbatch)
    k2 = eval_rhs(Input,nbatch,w+k1*dt/3.0)
    k3 = eval_rhs(Input,nbatch,w-k1*dt/3.0+k2*dt)
    k4 = eval_rhs(Input,nbatch,w+k1*dt-k2*dt+k3*dt)
    for ce,k in zip(c,[k1,k2,k3,k4]): 
        w += ce*k*dt
    return w
```

On évalue la RHS (Right hand side) equation, et on ajuste les paramètres du réseau à chaque dt: 
```
#@tf.function
    def eval_rhs(self,Input,nbatch,w=None):
        '''
        Evaluate the RHS of the following evolution equation of network parameters: 

                    dW/dt = gamma(t)
        
        A least square problem is solved to find the time derivative of network 
        parameters. 

        Input: coordinates of spatial collocation points
        nbatch: number of batches. The value of nbatch depends on the scale of 
                given problem and available memory. 
        '''

        if w is not None:
            wtmp = self.get_weights_np()
            self.set_weights_np(w)
        #Number of equations. Hardcoded
        Ju = [[]] + [[]]
        #Calculate the Jacobian on nbatch data sets.
        for x in range(int(len(Input)/nbatch)):
            JUV = self.eval_NN_grad(tf.reshape(Input[x*nbatch:(x+1)*nbatch,:],[nbatch,-1]))
            for J, indEq in zip(JUV, range(len(Ju))):
                indk = [i for i in range(len(J))][::2]
                indb = [i for i in range(len(J))][1::2]
                Jn = [j.numpy() for j in J]
                Jn = [jn.reshape(jn.shape[0],-1) for jn in Jn]
                Jn = np.concatenate(Jn,axis = 1)
                Ju[indEq] += [Jn]
        JJ = np.concatenate([np.concatenate(J, axis = 0) for J in Ju],axis = 0)

        dudt = self.rhs(self.output, Input, self.eq_params)
        dudt = np.concatenate([e.numpy().flatten() for e in dudt])

        # Calculate the time derivative of network weights
        sol = np.linalg.lstsq(JJ,dudt,rcond = 1e-3)
        
        dwdt = sol[0]

        if w is not None:
            self.set_weights_np(wtmp)

        return dwdt
```