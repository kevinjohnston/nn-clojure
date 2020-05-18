(ns nn-clojure.util
  "TODO create namespace doc"
  (:require
   [clojure.spec.alpha :as s]))

;;;;; macros
(defmacro vex [spec s] `(or (s/valid? ~spec ~s) (s/explain ~spec ~s)))
(defmacro with-out-str-data-map
  [& body]
  `(let [s# (new java.io.StringWriter)]
     (binding [*out* s#]
       (let [r# ~@body]
         {:result r#
          :str    (str s#)}))))

;;;;; helpers
(defn shift [x plus minus] (- (+ x plus) minus))
(defn remove-formatting
  [s]
  (clojure.string/replace s #"[\"\n]" ""))

;;;;; math fn
(defn sum
  ([l] (apply + l))
  ([x & xs] (+ x (sum xs))))
(defn avg [l] (/ (sum l) (count l)))
(def  mean avg)
(def  diff (comp #(Math/abs %) -))
(defn dot [v1 v2] (sum (map * v1 v2)))
(defn square [x] (Math/pow x 2))
(def  sqrt #(Math/sqrt %))
(defn limit-range [x high low] (-> x (max low) (min high)))
(defn variance [xs] (let [avg (mean xs)]
                      (mean (map (comp square diff)
                                 (repeat avg)
                                 xs))))
(defn std-dev [xs] (-> xs variance sqrt))
