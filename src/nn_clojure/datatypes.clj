(ns nn-clojure.datatypes
  "TODO create namespace doc"
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

;;;;; specs
;; define numbers excluding ##NaN ##Inf ##-Inf
(s/def ::number (s/and number? #(Double/isFinite %)))
(s/def ::even (s/with-gen even?
                #(s/gen (s/and int? (fn [i] (= 0 (mod i 2)))))))
(s/def ::pos (s/with-gen pos?
               #(s/gen (s/and ::number pos?))))
(s/def ::not-neg (s/with-gen >=0
                   #(s/gen (s/and ::number >=0))))

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

