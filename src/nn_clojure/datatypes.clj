(ns nn-clojure.datatypes
  "Define datatypes and, when necessary, generators.

  This file exists to give a clear specification of the otherwise typeless
  parameters used in functions throughout the program.

  These types fit into several major cateogries
  Non-domain types:
    - precursors -- these types define things that aren't specifically related
      to neural networks but get used by later types. these types tend to be
      more primitive than even domain primitives and get used to define domain
      types in generic programming level abstractions.
      example:
        ::pos - a positive finite integer
  
    - latteral -- these define types that are needed in the process of
      modeling the domain process but aren't strictly part of the domain these
      types tend to exist parallel to domain types or contain them but never be
      contained by them.
      example:
        ::ctx - a hashmap used to contain domain types and other information
          that gets passed through the modeled
  
  Domain types:
    - primitives -- these define atomic domain types that will be aggregated
      into larger domain types.
      example:
        ::weight -- an integer representing the strength of connection between
          two neurons
  
    - composites -- these define types that are strictly composed of
      primitives, other domain composites, or combinations of both (e.g.
      neuron, layer, etc)
      example:
        ::neuron -- a hashmap that contains ::weight, ::bias and at times other
          domain information."
      
  (:require
   [nn-clojure.util :refer :all]
   [clojure.spec.alpha :as s]
   [clojure.spec.gen.alpha :as gen]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Specs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; helpers
(def ^:dynamic assumed-max 10)
(def ^:dynamic assumed-min 0.0)
(defn- one? [n] (= n 1))
(defn- squash
  "Squash a number between 0.0 and 1.0 inclusive.

  Assumes inputs will tend to be between 0.0 and 10"
  [n]
  (/ (- (min (max n assumed-min) assumed-max) assumed-min)
     (- assumed-max assumed-min)))
(defn- >=0 [n] (>= n 0))
(defn- valid-number-of-weights
  [nn]
  (if (one? (count nn))
    true
    (let [last-layer (nth nn (-> nn count dec))
          next-to-last-layer (nth nn (-> nn count dec dec))
          neuron (first last-layer)]
      (and (= (-> neuron ::weights count)
              (-> next-to-last-layer count))
           (valid-number-of-weights (butlast nn))))))
(declare nn)
(declare ctx)
(declare gen)

;;;;; non-domain types
;;; precursors
;; define numbers excluding ##NaN ##Inf ##-Inf
(s/def ::number (s/and number? #(Double/isFinite %)))
(s/def ::even (s/with-gen even?
                #(s/gen (s/and int? (fn [i] (= 0 (mod i 2)))))))
(s/def ::pos (s/with-gen pos?
               #(s/gen (s/and ::number pos?))))
(s/def ::not-neg (s/with-gen >=0
                   #(s/gen (s/and ::number >=0))))
;;;;; domain types
;;; primitives
(s/def ::weight ::number)
(s/def ::weights (s/and (s/coll-of ::weight :kind vector?) not-empty))
(s/def ::bias ::number)
(s/def ::activation ::number)
(s/def :activation/layer (s/coll-of ::activation :kind vector?))
(s/def :activation/nn (s/coll-of :activation/layer :kind vector?))
(s/def :activation/sigmoid (s/with-gen #(<= 0.0 % 1.0)
                             #(gen/fmap squash (s/gen ::pos))))
(s/def :activation/fn-name #{::sigmoid})

(s/def ::input ::activation)
(s/def ::total-input ::number)
(s/def ::total-inputs (s/coll-of ::total-input :kind vector?))
(s/def ::total-inputs-nn (s/coll-of ::total-inputs :kind vector?))

(s/def ::goal ::activation)
(s/def ::goals (s/coll-of ::goal :kind vector?))

(s/def ::delta ::number)
(s/def :delta/layer (s/coll-of ::delta :kind vector?))
(s/def :delta/nn (s/coll-of :delta/layer :kind list?))

(s/def ::learning-rate ::pos)

;;; composites
(s/def ::neuron (s/keys :req [::weights ::bias]))
(s/def ::layer (s/and (s/coll-of ::neuron :kind vector?)
                      not-empty
                      #(apply = (map (fn [neuron] (-> neuron ::weights count))
                                     %))))

(s/def ::nn (s/with-gen (s/and (s/coll-of ::layer :kind vector?)
                               not-empty
                               valid-number-of-weights)
              #(gen/fmap nn (s/gen
                             (s/and (s/coll-of (s/and ::pos int?))
                                    (fn [l] (-> l count (>= 2))))))))

(s/def :adjustments/bias ::bias)
(s/def :adjustments/weight ::weight)
(s/def :adjustments/neuron ::neuron)
(s/def :adjustments/layer ::layer) ;; can fail

;; use custom generator since default is too slow
(s/def :adjustments/nn (s/with-gen (s/and (s/coll-of ::layer :kind list?)
                                          not-empty
                                          valid-number-of-weights)
                         #(gen/fmap (fn [l] (->> l nn (into () reverse)))
                                    (s/gen
                                     (s/and (s/coll-of (s/and ::pos int?))
                                            (fn [l] (-> l count (>= 2))))))))

(s/def ::seed ::number)

;; use custom generator since default is too slow
(s/def ::ctx (s/with-gen (s/and (s/keys :req [::nn
                                              ::learning-rate
                                              :activation/fn-name]
                                        :opt [::seed
                                              :delta/nn
                                              ::goals
                                              :activation/nn
                                              :adjustments/nn
                                              :train/i-train
                                              :train/max-train
                                              :train/target
                                              :train/examples])
                                #(or (-> % ::goals nil?)
                                     (= (-> % ::goals count)
                                        (-> % ::nn last count)))
                                #(-> % ::nn count (>= 2)))
               #(gen/fmap (fn [[inputs outputs]] (ctx inputs outputs))
                          (s/gen
                           (s/tuple (s/and ::pos int?)
                                    (s/and ::pos int?))))))

;; data types used when training
(s/def :train/domain (s/and ::number #(<= 0.0 % 1.0)))
(s/def :train/inputs (s/coll-of :train/domain))
(s/def :train/range-up (s/coll-of :train/domain))
(s/def :train/range-down (s/coll-of :train/domain))
(s/def :train/goal ::goals)
(s/def :train/pattern (s/keys :req [:train/inputs
                                    :train/range-up
                                    :train/range-down
                                    :train/goal]))
(s/def :train/ratio :train/domain)
(s/def :train/data-point (s/coll-of :train/domain :kind vector?))
(s/def :train/data-set map?)
(s/def :train/example (s/tuple :train/data-point
                               :train/goal
                               :train/pattern))
(s/def :train/examples (s/coll-of :train/example))
(s/def :train/tests :train/examples)
(s/def :train/split-data (s/keys :req [:train/examples :train/tests]))

(s/def :train/epoch (s/coll-of :train/examples))
(s/def :train/epochs (s/and ::number int?))
(s/def :train/batch-size (s/and ::number pos? int?))

(s/def :train/batch :train/examples)
(s/def :train/lesson :train/example)
(s/def :train/max-train (s/and ::number int?))
(s/def :train/i-train (s/and ::number int?))
(s/def :train/error map?)
(s/def :train/target :train/domain)

;; possible datatypes
(s/def :train/point (s/keys :req [:train/epoch
                                  :train/batch
                                  :train/example
                                  :train/epochs]))
(s/def :train/seq (s/coll-of :train/point :kind seq?))



;;;;; gain finer control over randomness (to make use of seeds)
(def ^:dynamic *r* (java.util.Random. 42))

(defn rand
  "Override clojure.core/rand in order to use a dynamically bindable Random
  generator."
  ([] (.nextDouble *r*))
  ([n] (* n (.nextDouble *r*))))

(defn repeatedly
  "Override clojure.core/repeatedly, original definition is declared static
  which prevents dynamic rebinding within the repeated function, the lazy
  sequence version similarly won't allow rebinding of values internal to the
  function."
  ([n f]
   (reduce (fn [acc _] (conj acc (f)))
           []
           (take n (repeat nil)))))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Initialization functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn neuron
  "Create a new neuron with a given number of inputs."
  [num-weights]
  {:post [(vex ::neuron %)]}
  ;; use rand instead of spec generators so that results are repeatable
  {::weights (-> num-weights
                 (repeatedly #(- (rand 2) 1))
                 (vec))
   ::bias    (- (rand 2) 1)})

(defn nn
  "Create a new neural network

  Input is a list where each number represents the number of neurons in that
  layer (including the input layer)."
  [neurons-per-layer]
  {:pre  [(vex (s/and (s/coll-of (s/and ::pos int?))
                      #(-> % count (>= 2)))
               neurons-per-layer)]
   :post [(vex ::nn %)]}
  (mapv (fn [inputs neurons] (vec (repeatedly neurons #(neuron inputs))))
        neurons-per-layer
        (next neurons-per-layer)))

(defn ctx
  "Create a top level data structure representing the state of the app."
  ([inputs outputs]
   (binding [*r* (java.util.Random. 42)]
     (ctx (flatten [inputs
                    (repeatedly (-> inputs (/ 500) int inc)
                                #(-> inputs (/ 50) int inc))
                    outputs])
          0.1
          42)))
  ([layers learning-rate seed]
   {:post [(vex ::ctx %)]}
   (binding [*r* (if seed
                   (java.util.Random. seed)
                   (java.util.Random.))]
     (let [ctx {::nn                 (nn layers)
                ::learning-rate      learning-rate
                :activation/fn-name  ::sigmoid}]
       (if seed
         (assoc ctx ::seed seed)
         ctx)))))
