(ns user
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [nn-clojure.core :refer :all]
            [nn-clojure.datatypes :as dt]
            [nn-clojure.domain :as do]
            [nn-clojure.train :as tr]
            [nn-clojure.util :refer :all]))

(declare try-it!)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; example nn and ctx
;;; for 2421 layout (2 inputs, 4 nodes in first hidden layer,
;;; 2 nodes in second hidden layer, 1 node in output layer)
(def xor-2421-nn
  [[#:nn-clojure.datatypes{:weights
                            [0.7 0.3],
                            :bias -0.3}
    #:nn-clojure.datatypes{:weights
                            [-0.6 0.4],
                            :bias 0.4}
    #:nn-clojure.datatypes{:weights
                            [0.5 -0.5],
                            :bias 0.5}
    #:nn-clojure.datatypes{:weights
                            [-0.4 -0.6],
                            :bias -0.6}]
   [#:nn-clojure.datatypes{:weights
                            [0.35
                             -0.38
                             -0.41
                             0.44],
                            :bias 0.4}
    #:nn-clojure.datatypes{:weights
                            [0.65
                             -0.62
                             -0.59
                             0.56],
                            :bias 0.6}]
   [#:nn-clojure.datatypes{:weights
                            [0.45 0.55],
                            :bias -0.5}]])

(def xor-2421-ctx
  {::dt/nn             xor-2421-nn
   ::dt/learning-rate  0.1,
   :activation/fn-name :nn-clojure.datatypes/sigmoid})

(def xor-00
  {:train/inputs [0 0]
   :train/range-down [0.0 0.0]
   :train/range-up [0.0 0.0]
   :train/goal [0]})

(def xor-01
  {:train/inputs [0 1]
   :train/range-down [0.0 0.0]
   :train/range-up [0.0 0]
   :train/goal [1]})

(def xor-10
  {:train/inputs [1 0]
   :train/range-down [0.0 0.0]
   :train/range-up [0.0 0.0]
   :train/goal [1]})

(def xor-11
  {:train/inputs [1 1]
   :train/range-down [0.0 0.0]
   :train/range-up [0.0 0]
   :train/goal [0]})

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Configuration examples
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; using a standard network show how different configuration options affect
;;;;; how it learns
(def config-ctx xor-2421-ctx)
(def config1 (-> config-ctx
                 (assoc ::dt/learning-rate  0.01
                        :activation/fn-name ::dt/sigmoid)
                 (tr/+config 4 0.00005)
                 (tr/+training-data 2 0.5 xor-00 xor-01 xor-10 xor-11)))
(def config2 (-> config-ctx
                 (assoc ::dt/learning-rate  1.5
                        :activation/fn-name ::dt/sigmoid)
                 (tr/+config 1 0.01)
                 (tr/+training-data 2 0.5 xor-00 xor-01 xor-10 xor-11)))
;;(def config3 (gen/config ctx))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Training Examples
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn single-pass-2421
  "Example of how context is changed when going through one iteration of forward
  and back propagation."
  []
  (-> xor-2421-ctx
      (assoc :activation/nn [[1 0]]
             ::dt/goals      [1])
      do/forward-propagation
      do/backward-propagation))

(defn test-ex
  [exemplar]
  [(-> exemplar (assoc :activation/nn [[0 0]] ::dt/goals [0])
       do/forward-propagation do/total-error)
   (-> exemplar (assoc :activation/nn [[0 1]] ::dt/goals [1])
       do/forward-propagation do/total-error)
   (-> exemplar (assoc :activation/nn [[1 0]] ::dt/goals [1])
       do/forward-propagation do/total-error)
   (-> exemplar (assoc :activation/nn [[1 1]] ::dt/goals [0])
       do/forward-propagation do/total-error)])


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Reductions test
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn setup
  ([] (setup (rand-int 1000)))
  ([seed]
  (binding [dt/*r* (java.util.Random. seed)]
  (let [rnd dt/*r*
        nn (dt/nn [2 4 1])
        learning-rate 1.5
        ;; 1.5 and 3000 epochs for < .01 error
        ;; .01 and 100000 epochs for < .00005% error

        initial-learning-rate 1.5
        constant 4
        ctx {::dt/nn nn
             ::dt/learning-rate learning-rate
             :activation/fn-name ::dt/sigmoid
             ::dt/rnd rnd
             ;;:train/annealing-fn (fn [x] (* 1.5 4 x))
             }
        size 2
        ratio 0.5
        pattern-00 {:train/inputs [0 0]
                    :train/range-down [0.0 0.0]
                    :train/range-up [0.0 0.0]
                    :train/goal [0]}
        pattern-01 {:train/inputs [0 1]
                    :train/range-down [0.0 0.0]
                    :train/range-up [0.0 0]
                    :train/goal [1]}
        pattern-10 {:train/inputs [1 0]
                    :train/range-down [0.0 0.0]
                    :train/range-up [0.0 0.0]
                    :train/goal [1]}
        pattern-11 {:train/inputs [1 1]
                    :train/range-down [0.0 0.0]
                    :train/range-up [0.0 0]
                    :train/goal [0]}
        batch-size 4
        target 0.01]

    (-> ctx
        (tr/+config batch-size target)
        (tr/+training-data size
                           ratio
                           pattern-00
                           pattern-01
                           pattern-10
                           pattern-11))))))

(defn xor-train
  ([_] (xor-train))
  ([]
   (let [eval-fn #(< (tr/max-err %) 0.1)
         ctx (merge (setup) {:train/eval-fn eval-fn
                             :train/target        0.01
                             :train/min-epochs    1
                             :train/max-epochs 4000})]
     (->> ctx
          train
          evaluate-training))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Repl exploratory functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn toggle-asserts
  "Toggle whether :pre and :post conditions get run."
  []
  (set! *assert* (not *assert*)))

(defn gen
  "Convenience function to generate an example of a given spec."
  [spec]
  (first (gen/sample (s/gen spec) 1)))

(defn try-it!
  []
  (let [num-threads (int (max (/ logical-threads 2) 1))]
    (println "Using" num-threads "thread(s) for training...")
    (pmap xor-train (range num-threads))))

