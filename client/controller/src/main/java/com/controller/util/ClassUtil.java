/*
 * OtterTune - ClassUtil.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.util;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.collections15.CollectionUtils;
import org.apache.commons.lang.ClassUtils;
import org.apache.log4j.Logger;

/** @author pavlo */
public abstract class ClassUtil {
  private static final Logger LOG = Logger.getLogger(ClassUtil.class);

  private static final Class<?>[] EMPTY_ARRAY = new Class[] {};

  private static final Map<Class<?>, List<Class<?>>> CACHE_getSuperClasses =
      new HashMap<Class<?>, List<Class<?>>>();
  private static final Map<Class<?>, Set<Class<?>>> CACHE_getInterfaceClasses =
      new HashMap<Class<?>, Set<Class<?>>>();

  /**
   * Check if the given object is an array (primitve or native).
   * http://www.java2s.com/Code/Java/Reflection/Checkifthegivenobjectisanarrayprimitveornative.htm
   *
   * @param obj Object to test.
   * @return True of the object is an array.
   */
  public static boolean isArray(final Object obj) {
    return (obj != null ? obj.getClass().isArray() : false);
  }

  public static boolean[] isArray(final Object[] objs) {
    boolean[] isArray = new boolean[objs.length];
    for (int i = 0; i < objs.length; i++) {
      isArray[i] = ClassUtil.isArray(objs[i]);
    } // FOR
    return (isArray);
  }

  /**
   * Convert a Enum array to a Field array This assumes that the name of each Enum element
   * corresponds to a data member in the clas
   *
   * @param <E>
   * @param clazz
   * @param members
   * @return
   * @throws NoSuchFieldException
   */
  public static <E extends Enum<?>> Field[] getFieldsFromMembersEnum(Class<?> clazz, E[] members)
      throws NoSuchFieldException {
    Field[] fields = new Field[members.length];
    for (int i = 0; i < members.length; i++) {
      fields[i] = clazz.getDeclaredField(members[i].name().toLowerCase());
    } // FOR
    return (fields);
  }

  /**
   * Get the generic types for the given field
   *
   * @param field
   * @return
   */
  public static List<Class<?>> getGenericTypes(Field field) {
    ArrayList<Class<?>> genericClasses = new ArrayList<Class<?>>();
    Type gtype = field.getGenericType();
    if (gtype instanceof ParameterizedType) {
      ParameterizedType ptype = (ParameterizedType) gtype;
      getGenericTypesImpl(ptype, genericClasses);
    }
    return (genericClasses);
  }

  private static void getGenericTypesImpl(ParameterizedType ptype, List<Class<?>> classes) {
    // list the actual type arguments
    for (Type t : ptype.getActualTypeArguments()) {
      if (t instanceof Class) {
        //                System.err.println("C: " + t);
        classes.add((Class<?>) t);
      } else if (t instanceof ParameterizedType) {
        ParameterizedType next = (ParameterizedType) t;
        //                System.err.println("PT: " + next);
        classes.add((Class<?>) next.getRawType());
        getGenericTypesImpl(next, classes);
      }
    } // FOR
    return;
  }

  /**
   * Return an ordered list of all the sub-classes for a given class Useful when dealing with
   * generics
   *
   * @param elementClass
   * @return
   */
  public static List<Class<?>> getSuperClasses(Class<?> elementClass) {
    List<Class<?>> ret = ClassUtil.CACHE_getSuperClasses.get(elementClass);
    if (ret == null) {
      ret = new ArrayList<Class<?>>();
      while (elementClass != null) {
        ret.add(elementClass);
        elementClass = elementClass.getSuperclass();
      } // WHILE
      ret = Collections.unmodifiableList(ret);
      ClassUtil.CACHE_getSuperClasses.put(elementClass, ret);
    }
    return (ret);
  }

  /**
   * Get a set of all of the interfaces that the element_class implements
   *
   * @param elementClass
   * @return
   */
  @SuppressWarnings("unchecked")
  public static Collection<Class<?>> getInterfaces(Class<?> elementClass) {
    Set<Class<?>> ret = ClassUtil.CACHE_getInterfaceClasses.get(elementClass);
    if (ret == null) {
      //            ret = new HashSet<Class<?>>();
      //            Queue<Class<?>> queue = new LinkedList<Class<?>>();
      //            queue.add(element_class);
      //            while (!queue.isEmpty()) {
      //                Class<?> current = queue.poll();
      //                for (Class<?> i : current.getInterfaces()) {
      //                    ret.add(i);
      //                    queue.add(i);
      //                } // FOR
      //            } // WHILE
      ret = new HashSet<Class<?>>(ClassUtils.getAllInterfaces(elementClass));
      if (elementClass.isInterface()) {
        ret.add(elementClass);
      }
      ret = Collections.unmodifiableSet(ret);
      ClassUtil.CACHE_getInterfaceClasses.put(elementClass, ret);
    }
    return (ret);
  }

  @SuppressWarnings("unchecked")
  public static <T> T newInstance(String className, Object[] params, Class<?>[] classes) {
    return ((T) ClassUtil.newInstance(ClassUtil.getClass(className), params, classes));
  }

  public static <T> T newInstance(Class<T> targetClass, Object[] params, Class<?>[] classes) {
    //        Class<?> const_params[] = new Class<?>[params.length];
    //        for (int i = 0; i < params.length; i++) {
    //            const_params[i] = params[i].getClass();
    //            System.err.println("[" + i + "] " + params[i] + " " + params[i].getClass());
    //        } // FOR

    Constructor<T> constructor = ClassUtil.getConstructor(targetClass, classes);
    T ret = null;
    try {
      ret = constructor.newInstance(params);
    } catch (Exception ex) {
      throw new RuntimeException(
          "Failed to create new instance of " + targetClass.getSimpleName(), ex);
    }
    return (ret);
  }

  /**
   * Create an object for the given class and initialize it from conf
   *
   * @param theClass class of which an object is created
   * @param expected the expected parent class or interface
   * @return a new object
   */
  public static <T> T newInstance(Class<?> theClass, Class<T> expected) {
    T result;
    try {
      if (!expected.isAssignableFrom(theClass)) {
        throw new Exception(
            "Specified class "
                + theClass.getName()
                + ""
                + "does not extend/implement "
                + expected.getName());
      }
      Class<? extends T> clazz = (Class<? extends T>) theClass;
      Constructor<? extends T> meth = clazz.getDeclaredConstructor(EMPTY_ARRAY);
      meth.setAccessible(true);
      result = meth.newInstance();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return result;
  }

  public static <T> T newInstance(String className, Class<T> expected)
      throws ClassNotFoundException {
    return newInstance(getClass(className), expected);
  }

  /**
   * @param <T>
   * @param targetClass
   * @param params
   * @return
   */
  @SuppressWarnings("unchecked")
  public static <T> Constructor<T> getConstructor(Class<T> targetClass, Class<?>... params) {
    NoSuchMethodException error = null;
    try {
      return (targetClass.getConstructor(params));
    } catch (NoSuchMethodException ex) {
      // The first time we get this it can be ignored
      // We'll try to be nice and find a match for them
      error = ex;
    }
    assert (error != null);

    if (LOG.isDebugEnabled()) {
      LOG.debug("TARGET CLASS:  " + targetClass);
      LOG.debug("TARGET PARAMS: " + Arrays.toString(params));
    }

    List<Class<?>>[] paramSuper = (List<Class<?>>[]) new List[params.length];
    for (int i = 0; i < params.length; i++) {
      paramSuper[i] = ClassUtil.getSuperClasses(params[i]);
      if (LOG.isDebugEnabled()) {
        LOG.debug("  SUPER[" + params[i].getSimpleName() + "] => " + paramSuper[i]);
      }
    } // FOR

    for (Constructor<?> c : targetClass.getConstructors()) {
      Class<?>[] ctypes = c.getParameterTypes();
      if (LOG.isDebugEnabled()) {
        LOG.debug("CANDIDATE: " + c);
        LOG.debug("CANDIDATE PARAMS: " + Arrays.toString(ctypes));
      }
      if (params.length != ctypes.length) {
        continue;
      }

      for (int i = 0; i < params.length; i++) {
        List<Class<?>> csuper = ClassUtil.getSuperClasses(ctypes[i]);
        if (LOG.isDebugEnabled()) {
          LOG.debug("  SUPER[" + ctypes[i].getSimpleName() + "] => " + csuper);
        }
        if (CollectionUtils.intersection(paramSuper[i], csuper).isEmpty() == false) {
          return ((Constructor<T>) c);
        }
      } // FOR (param)
    } // FOR (constructors)
    throw new RuntimeException(
        "Failed to retrieve constructor for " + targetClass.getSimpleName(), error);
  }

  /**
   * @param className
   * @return
   */
  public static Class<?> getClass(String className) {
    Class<?> targetClass = null;
    try {
      ClassLoader loader = ClassLoader.getSystemClassLoader();
      targetClass = (Class<?>) loader.loadClass(className);
    } catch (Exception ex) {
      throw new RuntimeException("Failed to retrieve class for " + className, ex);
    }
    return (targetClass);
  }

  /**
   * Returns true if asserts are enabled. This assumes that we're always using the default system
   * ClassLoader
   */
  public static boolean isAssertsEnabled() {
    boolean ret = false;
    try {
      assert (false);
    } catch (AssertionError ex) {
      ret = true;
    }
    return (ret);
  }
}
