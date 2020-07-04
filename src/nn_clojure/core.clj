(ns nn-clojure.core
  "TODO namespace documentation"
  (:gen-class)
  (:require
   [clojure.spec.alpha :as s]
   [nn-clojure.datatypes :as dt]
   [nn-clojure.domain :as do]
   [nn-clojure.train :as tr]
   [nn-clojure.util :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def logical-threads (. (Runtime/getRuntime) availableProcessors))

(def size "The number of data points to generated from each pattern" 2)
(def ratio "Ratio of generated data held back for testing" 0.5)
(def batch-size "The number of data points used in each training batch" 4)
(def target "The target error rate" 0.01)

(def conf-quick "Merge with a ctx when training a neural network for quick and dirty results."
  {::dt/learning-rate        1.5
   :train/max-epochs      4000
   :train/min-epochs         1
   :train/batch-size         4
   :train/target             0.01})
(def conf-accurate "Merge with a ctx when training a neural network for accurate results."
  {::dt/learning-rate        0.01
   :train/max-epochs  10000000
   :train/min-epochs         1
   :train/batch-size         4
   :train/target             0.00005})

(defn ctx
  "Return a new ctx including a new, randomly generated, neural network to run."
  ([] (ctx [2 4 1]))
  ([layers]
   {::dt/nn             (dt/nn layers)
    ::dt/learning-rate  nil
    :activation/fn-name ::dt/sigmoid
    :train/max-epochs   nil
    :train/min-epochs   nil
    :train/buffer       1.0
    :train/eval-fn      #(< (tr/max-err %) 0.1)
    ::dt/rnd            dt/*r*
    :train/batch-size   4
    :train/target       nil
    :train/max-raw-data 10}))

;;;;; training data
;;; xor
(def xor-pattern-00 {:train/inputs     [0 0]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0.0]
                     :train/goal       [0]})
(def xor-pattern-01 {:train/inputs     [0 1]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0]
                     :train/goal       [1]})
(def xor-pattern-10 {:train/inputs     [1 0]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0.0]
                     :train/goal       [1]})
(def xor-pattern-11 {:train/inputs     [1 1]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0]
                     :train/goal       [0]})

(def +xor-training-data
  (fn [ctx]
    (-> ctx
        (tr/+training-data size
                           ratio
                           xor-pattern-00
                           xor-pattern-01
                           xor-pattern-10
                           xor-pattern-11))))
;;; and
(def and-pattern-00 {:train/inputs     [0 0]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0.0]
                     :train/goal       [0]})
(def and-pattern-01 {:train/inputs     [0 1]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0]
                     :train/goal       [0]})
(def and-pattern-10 {:train/inputs     [1 0]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0.0]
                     :train/goal       [0]})
(def and-pattern-11 {:train/inputs     [1 1]
                     :train/range-down [0.0 0.0]
                     :train/range-up   [0.0 0]
                     :train/goal       [1]})

(def +and-training-data
  (fn [ctx]
    (-> ctx
        (tr/+training-data size
                           ratio
                           and-pattern-00
                           and-pattern-01
                           and-pattern-10
                           and-pattern-11))))
;;; or
(def or-pattern-00 {:train/inputs     [0 0]
                    :train/range-down [0.0 0.0]
                    :train/range-up   [0.0 0.0]
                    :train/goal       [0]})
(def or-pattern-01 {:train/inputs     [0 1]
                    :train/range-down [0.0 0.0]
                    :train/range-up   [0.0 0]
                    :train/goal       [1]})
(def or-pattern-10 {:train/inputs     [1 0]
                    :train/range-down [0.0 0.0]
                    :train/range-up   [0.0 0.0]
                    :train/goal       [1]})
(def or-pattern-11 {:train/inputs     [1 1]
                    :train/range-down [0.0 0.0]
                    :train/range-up   [0.0 0]
                    :train/goal       [1]})

(def +or-training-data
  (fn [ctx]
    (-> ctx
        (tr/+training-data size
                           ratio
                           or-pattern-00
                           or-pattern-01
                           or-pattern-10
                           or-pattern-11))))
;;; nand
(def nand-pattern-00 {:train/inputs     [0 0]
                      :train/range-down [0.0 0.0]
                      :train/range-up   [0.0 0.0]
                      :train/goal       [1]})
(def nand-pattern-01 {:train/inputs     [0 1]
                      :train/range-down [0.0 0.0]
                      :train/range-up   [0.0 0]
                      :train/goal       [1]})
(def nand-pattern-10 {:train/inputs     [1 0]
                      :train/range-down [0.0 0.0]
                      :train/range-up   [0.0 0.0]
                      :train/goal       [1]})
(def nand-pattern-11 {:train/inputs     [1 1]
                      :train/range-down [0.0 0.0]
                      :train/range-up   [0.0 0]
                      :train/goal       [0]})

(def +nand-training-data
  (fn [ctx]
    (-> ctx
        (tr/+training-data size
                           ratio
                           nand-pattern-00
                           nand-pattern-01
                           nand-pattern-10
                           nand-pattern-11))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn -main
  [& args]
  (let [xor-ctx  (+xor-training-data  (merge (ctx) conf-quick))
        and-ctx  (+and-training-data  (merge (ctx) conf-quick))
        nand-ctx (+nand-training-data (merge (ctx) conf-quick))
        or-ctx   (+or-training-data   (merge (ctx) conf-quick))
        train-it (fn [[logic-type ctx]]
                   (println (str logic-type
                                 (-> ctx
                                     tr/train
                                     tr/evaluate-training))))]
    (pmap train-it [["XOR "  xor-ctx]
                    ["NAND " nand-ctx]
                    ["AND "  and-ctx]
                    ["OR "   or-ctx]])
    ;; shutdown pmap threads as soon as they complete
    (shutdown-agents)))
