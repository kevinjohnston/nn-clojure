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

