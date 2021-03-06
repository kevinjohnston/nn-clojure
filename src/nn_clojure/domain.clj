(ns nn-clojure.domain
  "Provide domain logic for forward and back propagation.

  Forward propagation is the process of calculating the output of a neural
  network. This requires calculating the ouput for each layer, which in turn
  requires calculating the output of each neuron. These outputs will need to
  be stored for back propagation to proceed.

  Back propagation is the process of adjusting the weights and bias of each
  neuron in the neural network in order to reduce the difference between a
  neural networks output and the desired output. This can be though of as
  taking a step in a large multi-dimensional space (aka gradient descent).

  Back propagation uses multi-dimensional calculus to calculate the how much
  each weight and bias of each neuron contributes to the final error. Partial
  derivative functions are named to reflect their calculus representations.
  e.g. (defn dE|dA ....) this function is the partial derivative (d) of the
  total error (E) with respect to (|) the partial derivative (d) of a neurons
  output (A).
    - math symbols used
      A -- the output of a nueron after applying the activation function to the
           neurons total inputs
      B -- the bias of a neuron
      E -- the error calculated for a specific run
      W -- the weight of a neuron
      Z -- the sum total inputs to a neuron after adjusting for weight and bias
    - subscripts and indicators
      l -- subscript showing that a math symbol is referring to a node in a
           hidden layer (rather than an output layer). note: input layers don't
           need to have their weights and biases calculated since they are
           modeled as simple numbers.
      + -- used to indicate that the value results from a sum of it's partials
           e.g. (defn dE+|dAl ....) this functions sums up every dE|dAl in a
           layer

  see 3blue1brown video for better understanding of neural networks
    https://www.youtube.com/watch?v=aircAruvnKk"
  (:require
   [clojure.spec.alpha :as s]
   [nn-clojure.datatypes :as dt]
   [nn-clojure.util :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def _ 0)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Accessor functions -- return data from ctx
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Accessors
(defn- goal
  "Return the goal of jth neuron in the output layer"
  [ctx j]
  {:pre  [(vex ::dt/goals (-> ctx ::dt/goals))]
   :post [(vex ::dt/goal %)]}
  (-> ctx ::dt/goals (nth j)))
(defn- output
  "Return the output of jth neuron in ith layer"
  [ctx i j]
  {:pre  [(vex :activation/nn (-> ctx :activation/nn))]
   :post [(vex ::dt/activation %)]}
  (-> ctx :activation/nn (nth i) (nth j)))
(defn- input
  ([ctx i j k] (output (dec i) k)))
(defn- total-input
  "Return biased input for jth neuron in ith layer"
  [ctx i j]
  {:pre  [(vex ::dt/total-inputs-nn (-> ctx ::dt/total-inputs-nn))]
   :post [(vex ::dt/total-input %)]}
  ;; note i is decrimented since ::total-inputs-nn doesn't include input layer
  (-> ctx ::dt/total-inputs-nn (nth (dec i)) (nth j)))
(defn- layer
  "Return collection of neurons in ith layer"
  [ctx i]
  {:pre  [(vex ::dt/nn (-> ctx ::dt/nn))]
   :post [(vex ::dt/layer %)]}
  ;; note i is decrimented since ::dt/nn doesn't include input layer
  (-> ctx ::dt/nn (nth (dec i))))
(defn- bias
  "Return the bias of the jth neuron in the ith layer"
  [ctx i j]
  {:pre  [(vex ::dt/ctx ctx)]
   :post [(vex ::dt/bias %)]}
  (-> ctx (layer i) (nth j) ::dt/bias))
(defn- weight
  "Return the kth weight of the jth neuron in the ith layer"
  [ctx i j k]
  {:pre  [(vex ::dt/weights (-> ctx (layer i) (nth j) ::dt/weights))]
   :post [(vex ::dt/weight %)]}
  (-> ctx (layer i) (nth j) ::dt/weights (nth k)))
(declare num-deltas num-layers)
(defn- delta
  "Return the delta of the jth neuron in the ith layer"
  [ctx i j]
  {:pre  [(vex :delta/nn (-> ctx :delta/nn))]
   :post [(vex ::delta %)]}
  (let [i (shift i (num-deltas ctx) (num-layers ctx))]
    (-> ctx :delta/nn (nth i) (nth j))))
(defn- layer-deltas
  "Return the collection of deltas for the ith layer"
  [ctx i]
  {:pre  [(vex :delta/nn (-> ctx :delta/nn))]
   :post [(vex :delta/layer %)]}
  (-> ctx :delta/nn (nth (shift i (num-deltas ctx) (num-layers ctx)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Domain queries -- higher level questions about ctx
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- num-layer-inputs
  "Return the number of inputs to a neuron in the given layer."
  [ctx i]
  (let [any-neuron first]
    (-> ctx (layer i) any-neuron ::dt/weights count)))
(defn- num-deltas [ctx] (-> ctx :delta/nn count))
(defn- num-layers [ctx] (-> ctx :activation/nn count))
(defn- num-layer-deltas [ctx i] (count (layer-deltas ctx i)))
(defn- num-neurons-in-layer [{:keys [:activation/nn]} i] (-> nn (nth i) count))
(defn- last-layer-index [ctx] (-> ctx num-layers dec))
(defn- weights-from-neuron
  "Returns a vector of weights *from* the jth neuron in the ith layer."
  [ctx i j k]
  (let [neuron-layer (layer ctx (inc i))
        weights      (mapv #(get-in % [::dt/weights j]) neuron-layer)]
    weights))
(defn- computing-output-layer? [ctx i] (= (last-layer-index ctx) i))
(defn- type-of-layer
  [ctx i]
  {:pre  [(<= 0 i (last-layer-index ctx))]}
  (cond
    (= i 0)                         ::input
    (computing-output-layer? ctx i) ::output
    :default                        ::hidden))

(defn- error
  "Measurement of the difference between a neurons output and it's goal."
  [goal output]
  (* 1/2 (square (- goal output))))

(defmulti activation
  "Calc the activation of a neuron.

  Interesting to note that the activation function *must* be non-linear in order
  for training to be possible (on non-linear data)."
  (fn [type _] type))
(defmulti dActivation
  "Calc the derivative of the given activation fn with respect to it's input
  i.e. the 'total-input'."
  (fn [type _] type))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Math functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Helpers
(defn- chain
  "Chain rule for partial derivatives"
  [& partials]
  (fn [ctx i j k]
    (->> ((apply juxt partials) ctx i j k)
         (reduce *))))

;;;;; Partial derivatives
;;; output layer
(defn- dE|dA   [ctx i j k] (* -1 (- (goal ctx j) (output ctx i j))))
(defn- dA|dZ   [ctx i j k]
  (dActivation (:activation/fn-name ctx) (total-input ctx i j)))
(defn- dZ|dW   [ctx i j k] (output ctx (dec i) k))
(defn- dZ|dB   [ctx i j k] 1)
(defn- dE|dB   [ctx i j k] ((chain dE|dA dA|dZ dZ|dB) ctx i j k))
(defn- dE|dW   [ctx i j k] ((chain dE|dA dA|dZ dZ|dW) ctx i j k))

;;; hidden layer
(defn- dE|dZ   [ctx i j k]
  (case (type-of-layer ctx i)
    ::output (* (dE|dA ctx i j k)
                (dA|dZ ctx i j k))
    ::hidden (let [next-layer-deltas  (layer-deltas ctx (inc i))
                   next-layer-weights (weights-from-neuron ctx i j k)]
               (dot next-layer-deltas next-layer-weights))))
(defn- dZ|dAl  [ctx i j k] (weight ctx i j k))
(defn- dE+|dAl [ctx i j k]
  ;; because weights are stored on the neurons recieving inputs we need
  ;; to "switch perspective" to the next layer (i.e. inc i) and gather
  ;; the relevant weight from each of those neurons
  (let [i  (inc i)
        js (range (num-neurons-in-layer ctx i))
        k  j]
    (dot (mapv #(dE|dZ  ctx i % k) js)
         (mapv #(dZ|dAl ctx i % k) js))))
(defn- dAl|dZl [ctx i j k] (dA|dZ ctx i j k))
(defn- dZl|dWl [ctx i j k] (dZ|dW ctx i j k))
(defn- dZl|dBl [ctx i j k] (dZ|dB ctx i j k))
(defn- dE+|dWl [ctx i j k] ((chain dE+|dAl dAl|dZl dZl|dWl) ctx i j k))
(defn- dE+|dBl [ctx i j k] ((chain dE+|dAl dAl|dZl dZl|dBl) ctx i j k))
(defn- sigmoid
  "e^x / (e^x + 1)"
  [x]
  {:pre  [(vex ::dt/total-input x)]
   :post [(vex :activation/sigmoid %)]}
  (/ (Math/pow Math/E x)
     (+ 1 (Math/pow Math/E x))))
(defn- dSigmoid
  "sigmoid(x) * {1 - sigmoid(x)}"
  [x]
  {:pre  [(vex ::dt/total-input x)]
   :post [(vex ::dt/number %)]}
  (* (sigmoid x)
     (- 1 (sigmoid x))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Forward propagation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Helper functions
(defn- push-layer-results
  [ctx {:keys [:activation/layer ::dt/total-inputs] :as fire-layer-results}]
  (-> ctx
      (update :activation/nn conj layer)
      (update ::dt/total-inputs-nn conj total-inputs)))

;;;;; Activation functions and derivatives
(defmethod  activation ::dt/sigmoid [_ total-input]  (sigmoid total-input))
(defmethod dActivation ::dt/sigmoid [_ total-input] (dSigmoid total-input))

;;;;; Domain functions
(defn- fire
  "Fire a neuron."
  [{::dt/keys [weights bias] :as neuron} inputs act-name]
  {:pre  [(vex ::dt/neuron neuron)
          (vex :activation/layer inputs)
          (vex :activation/fn-name act-name)]
   :post [(vex ::dt/total-input (::dt/total-input %))
          (vex ::dt/activation (::dt/activation %))]}
  (let [total-input (+ (dot weights inputs) bias)]
    {::dt/total-input total-input
     ::dt/activation   (activation act-name total-input)}))

(defn- fire-layer
  "Fire a layer of neurons."
  [act-name inputs neurons]
  {:pre  [(vex ::dt/layer neurons)
          (vex :activation/layer inputs)
          (vex :activation/fn-name act-name)]
   :post [(vex ::dt/total-inputs (::dt/total-inputs %))
          (vex :activation/layer (:activation/layer %))]}
  (let [fire #(fire %1 %2 act-name)]
    (reduce (fn [m [neuron inputs]]
              (let [{::dt/keys [total-input activation]} (fire neuron inputs)]
                (-> m
                    (update ::dt/total-inputs conj total-input)
                    (update :activation/layer conj activation))))
            {::dt/total-inputs [] :activation/layer []}
            (map vector neurons (repeat inputs)))))

;;;;; Public functions
(declare forward-propagation)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Back propagation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Domain functions
(defn- bias-adj
  [{:keys [::dt/learning-rate] :as ctx} i j k]
  {:pre  [(vex ::dt/ctx ctx)
          (<= 0 i (num-layers ctx))
          (<= 0 j (num-neurons-in-layer ctx i))
          (<= 0 k (num-neurons-in-layer ctx (dec i)))]
   :post [(vex :adjustments/bias %)]}
  (case (type-of-layer ctx i)
    ::input  (throw (ex-info (str "Attempted to calculate bias adjustment for"
                                  " input layer (inputs have no preceeding"
                                  " layer for a bias to influence).")
                             {:ctx ctx}))
    ::output (limit-range (- (bias ctx i j)
                             (* learning-rate (dE|dB ctx i j k)))
                          1.0 -1.0)
    ::hidden (limit-range (- (bias ctx i j)
                             (* learning-rate (dE+|dBl ctx i j k)))
                          1.0 -1.0)))

(defn- weight-adj
  [{:keys [::dt/learning-rate :activation/nn] :as ctx} i j k]
  {:pre  [(vex ::dt/ctx ctx)
          (<= 0 i (num-layers ctx))
          (<= 0 j (num-neurons-in-layer ctx i))
          (<= 0 k (num-neurons-in-layer ctx (dec i)))]
   :post [(vex :adjustments/weight %)]}
  (case (type-of-layer ctx i)
    ::input  (throw (ex-info (str "Attempted to calculate weight adjustment for"
                                  " input layer (inputs have no preceeding"
                                  " layer to adjust the weights of).")
                             {:ctx ctx}))
    ::output (- (weight ctx i j k)
                (* learning-rate (dE|dW ctx i j k)))
    ::hidden (- (weight ctx i j k)
                (* learning-rate (dE+|dWl ctx i j k)))))

(defn- layer-adjustments
  [ctx i]
  {:pre  [(vex ::dt/ctx ctx) (<= 1 i (num-layers ctx))]
   :post [(vex :adjustments/layer %)]}
  (let [num-cur-layer-neurons  (num-neurons-in-layer ctx i)
        num-prev-layer-neurons (num-neurons-in-layer ctx (dec i))

        ;; collect weight adjustments
        weight-adjustments     (for [j (range num-cur-layer-neurons)]
                                 ;; get weight adj vector for each neuron
                                 {::dt/weights
                                  (vec (for [k (range num-prev-layer-neurons)]
                                         ;; get weight adj for each neuron pair
                                         (weight-adj ctx i j k)))})
        ;; collect bias adjustments
        bias-adjustments       (for [j (range num-cur-layer-neurons)]
                                 {::dt/bias (bias-adj ctx i j _)})]
    (mapv merge weight-adjustments bias-adjustments)))

(defn- deltas
  [ctx i]
  {:pre  [(vex ::dt/ctx ctx) (<= 0 i (num-layers ctx))]
   :post [(vex :delta/layer %)]}
  (let [delta-rule (case (type-of-layer ctx i)
                     ::output dE|dZ
                     ::hidden (chain dE+|dAl dAl|dZl))]
    (mapv delta-rule
          (repeat ctx)
          (repeat i)
          (range (num-neurons-in-layer ctx i))
          (repeat _))))

(defn- record-layer-step
  [ctx i]
  (-> ctx
      (update :adjustments/nn conj (layer-adjustments ctx i))
      (update :delta/nn       conj (deltas ctx i))))

;;;;; Public functions
(declare backward-propagation)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Public API
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn forward-propagation
  "Fire all neurons in the network and record activations and total inputs
  along the way."
  [{layers      ::dt/nn
    act-fn-name :activation/fn-name
    activations :activation/nn :as ctx}]
  {:pre  [(vex ::dt/ctx ctx)
          (vex :activation/nn activations)]
   :post [(vex ::dt/ctx %)]}
  (reduce (fn [ctx layer]
            (->> layer
                 (fire-layer act-fn-name (-> ctx :activation/nn peek))
                 (push-layer-results ctx)))
          (assoc ctx ::dt/total-inputs-nn [])
          layers))

(defn backward-propagation
  "Evaluate the network's outputs with respect to its goals recording the new
  weights and biased to be assigned if updating to the next step along a
  gradient descent."
  [ctx]
  {:pre  [(vex ::dt/ctx ctx)
          (vex ::dt/goals (-> ctx ::dt/goals))
          (vex ::dt/learning-rate (-> ctx ::dt/learning-rate))]
   :post [(vex ::dt/ctx %)]}
  (reduce record-layer-step
          ctx
          (-> ctx num-layers range next reverse)))

(defn total-error
  "Measurement of the error for all output neurons."
  [{:keys [::dt/goals :activation/nn] :as ctx}]
  {:pre  [(vex ::dt/ctx ctx)]
   :post [(vex ::dt/number %)]}
  (sum (map error goals (last nn))))
