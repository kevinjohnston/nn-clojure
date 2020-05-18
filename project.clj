(defproject nn-clojure "0.1.0-SNAPSHOT"
  :description "Example neural network (nn) written in clojure."
  :url "http://example.com/FIXME"
  :license {:name "MIT License"
            :url "none"
            :year 2020
            :key "mit"}
  :dependencies [[org.clojure/clojure "1.10.1"]]
  :main ^:skip-aot nn-clojure.core
  ;; give a reasonable timeout
  :repl-options {:timeout 5000}
  :target-path "target/%s"
  :profiles {:dev {:dependencies [[org.clojure/test.check "1.0.0"]]}
             :uberjar {:aot :all}}
  ;; license management plugin
  :plugins [[lein-license "0.1.8"]]
  ;; turn off asserts (e.g. :pre and :post conditions) by default to speed up
  ;; runtime
  :global-vars {*assert* false})
