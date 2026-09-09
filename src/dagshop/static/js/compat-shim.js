/**
 * Minimal `window._` shim providing only `.memoize`/`.throttle`.
 *
 * cytoscape-edgehandles' browser-global UMD build reads
 * `root["_"]["memoize"]`/`root["_"]["throttle"]` off a global lodash
 * object (see static/vendor/VENDOR.md). Rather than vendor all of
 * lodash for two functions, this is a small, dependency-free
 * reimplementation of just those two -- our own code, not a
 * repackaged third-party library.
 *
 * Not a general-purpose lodash replacement: do not rely on this for
 * anything beyond what cytoscape-edgehandles needs.
 */
(function (global) {
  "use strict";

  function memoize(fn, resolver) {
    var cache = new Map();
    var memoized = function () {
      var args = Array.prototype.slice.call(arguments);
      var key = resolver ? resolver.apply(this, args) : args[0];
      if (cache.has(key)) {
        return cache.get(key);
      }
      var result = fn.apply(this, args);
      cache.set(key, result);
      return result;
    };
    memoized.cache = cache;
    return memoized;
  }

  function throttle(fn, wait) {
    wait = typeof wait === "number" ? wait : 0;
    var lastCallAt = 0;
    var pendingTimer = null;
    var pendingArgs = null;
    var pendingThis = null;

    function invoke() {
      lastCallAt = Date.now();
      pendingTimer = null;
      fn.apply(pendingThis, pendingArgs);
      pendingArgs = null;
      pendingThis = null;
    }

    var throttled = function () {
      var now = Date.now();
      var remaining = wait - (now - lastCallAt);
      pendingArgs = arguments;
      pendingThis = this;
      if (remaining <= 0) {
        if (pendingTimer) {
          clearTimeout(pendingTimer);
          pendingTimer = null;
        }
        invoke();
      } else if (!pendingTimer) {
        pendingTimer = setTimeout(invoke, remaining);
      }
    };
    return throttled;
  }

  global._ = global._ || {};
  global._.memoize = memoize;
  global._.throttle = throttle;
})(window);
