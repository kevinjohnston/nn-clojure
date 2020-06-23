(ns nn-clojure.domain
  "TODO create namespace doc"
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

