/******************************************************************************
 *  Copyright 2015 by OLTPBenchmark Project                                   *
 *                                                                            *
 *  Licensed under the Apache License, Version 2.0 (the "License");           *
 *  you may not use this file except in compliance with the License.          *
 *  You may obtain a copy of the License at                                   *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 ******************************************************************************/


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

/**
 * 
 * @author pavlo
 *
 */
public abstract class ClassUtil {
    private static final Logger LOG = Logger.getLogger(ClassUtil.class);
    
    private static final Class<?>[] EMPTY_ARRAY = new Class[]{};
    
    private static final Map<Class<?>, List<Class<?>>> CACHE_getSuperClasses = new HashMap<Class<?>, List<Class<?>>>(); 
    private static final Map<Class<?>, Set<Class<?>>> CACHE_getInterfaceClasses = new HashMap<Class<?>, Set<Class<?>>>();

    /**
     * Check if the given object is an array (primitve or native).
     * http://www.java2s.com/Code/Java/Reflection/Checkifthegivenobjectisanarrayprimitveornative.htm
     * @param obj  Object to test.
     * @return     True of the object is an array.
     */
    public static boolean isArray(final Object obj) {
        return (obj != null ? obj.getClass().isArray() : false);
    }
    
    public static boolean[] isArray(final Object objs[]) {
        boolean is_array[] = new boolean[objs.length];
        for (int i = 0; i < objs.length; i++) {
            is_array[i] = ClassUtil.isArray(objs[i]);
        } // FOR
        return (is_array);
    }
    
    /**
     * Convert a Enum array to a Field array
     * This assumes that the name of each Enum element corresponds to a data member in the clas
     * @param <E>
     * @param clazz
     * @param members
     * @return
     * @throws NoSuchFieldException
     */
    public static <E extends Enum<?>> Field[] getFieldsFromMembersEnum(Class<?> clazz, E members[]) throws NoSuchFieldException {
        Field fields[] = new Field[members.length];
        for (int i = 0; i < members.length; i++) {
            fields[i] = clazz.getDeclaredField(members[i].name().toLowerCase());
        } // FOR
        return (fields);
    }

    /**
     * Get the generic types for the given field
     * @param field
     * @return
     */
    public static List<Class<?>> getGenericTypes(Field field) {
        ArrayList<Class<?>> generic_classes = new ArrayList<Class<?>>();
        Type gtype = field.getGenericType();
        if (gtype instanceof ParameterizedType) {
            ParameterizedType ptype = (ParameterizedType)gtype;
            getGenericTypesImpl(ptype, generic_classes);
        }
        return (generic_classes);
    }
        
    private static void getGenericTypesImpl(ParameterizedType ptype, List<Class<?>> classes) {
        // list the actual type arguments
        for (Type t : ptype.getActualTypeArguments()) {
            if (t instanceof Class) {
//                System.err.println("C: " + t);
                classes.add((Class<?>)t);
            } else if (t instanceof ParameterizedType) {
                ParameterizedType next = (ParameterizedType)t;
//                System.err.println("PT: " + next);
                classes.add((Class<?>)next.getRawType());
                getGenericTypesImpl(next, classes);
            }
        } // FOR
        return;
    }
    
    /**
     * Return an ordered list of all the sub-classes for a given class
     * Useful when dealing with generics
     * @param element_class
     * @return
     */
    public static List<Class<?>> getSuperClasses(Class<?> element_class) {
        List<Class<?>> ret = ClassUtil.CACHE_getSuperClasses.get(element_class);
        if (ret == null) {
            ret = new ArrayList<Class<?>>();
            while (element_class != null) {
                ret.add(element_class);
                element_class = element_class.getSuperclass();
            } // WHILE
            ret = Collections.unmodifiableList(ret);
            ClassUtil.CACHE_getSuperClasses.put(element_class, ret);
        }
        return (ret);
    }
    
    /**
     * Get a set of all of the interfaces that the element_class implements
     * @param element_class
     * @return
     */
    @SuppressWarnings("unchecked")
    public static Collection<Class<?>> getInterfaces(Class<?> element_class) {
        Set<Class<?>> ret = ClassUtil.CACHE_getInterfaceClasses.get(element_class);
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
            ret = new HashSet<Class<?>>(ClassUtils.getAllInterfaces(element_class));
            if (element_class.isInterface()) ret.add(element_class);
            ret = Collections.unmodifiableSet(ret);
            ClassUtil.CACHE_getInterfaceClasses.put(element_class, ret);
        }
        return (ret);
    }
    
    @SuppressWarnings("unchecked")
    public static <T> T newInstance(String class_name, Object params[], Class<?> classes[]) {
        return ((T)ClassUtil.newInstance(ClassUtil.getClass(class_name), params, classes));
    }

    
    public static <T> T newInstance(Class<T> target_class, Object params[], Class<?> classes[]) {
//        Class<?> const_params[] = new Class<?>[params.length];
//        for (int i = 0; i < params.length; i++) {
//            const_params[i] = params[i].getClass();
//            System.err.println("[" + i + "] " + params[i] + " " + params[i].getClass());
//        } // FOR
        
        Constructor<T> constructor = ClassUtil.getConstructor(target_class, classes);
        T ret = null;
        try {
            ret = constructor.newInstance(params);
        } catch (Exception ex) {
            throw new RuntimeException("Failed to create new instance of " + target_class.getSimpleName(), ex);
        }
        return (ret);
    }
    
    /**
     * 
     * @param <T>
     * @param target_class
     * @param params
     * @return
     */
    @SuppressWarnings("unchecked")
    public static <T> Constructor<T> getConstructor(Class<T> target_class, Class<?>...params) {
        NoSuchMethodException error = null;
        try {
            return (target_class.getConstructor(params));
        } catch (NoSuchMethodException ex) {
            // The first time we get this it can be ignored
            // We'll try to be nice and find a match for them
            error = ex;
        }
        assert(error != null);
        
        if (LOG.isDebugEnabled()) {
            LOG.debug("TARGET CLASS:  " + target_class);
            LOG.debug("TARGET PARAMS: " + Arrays.toString(params));
        }
        
        List<Class<?>> paramSuper[] = (List<Class<?>>[])new List[params.length]; 
        for (int i = 0; i < params.length; i++) {
            paramSuper[i] = ClassUtil.getSuperClasses(params[i]);
            if (LOG.isDebugEnabled()) LOG.debug("  SUPER[" + params[i].getSimpleName() + "] => " + paramSuper[i]);
        } // FOR
        
        for (Constructor<?> c : target_class.getConstructors()) {
            Class<?> cTypes[] = c.getParameterTypes();
            if (LOG.isDebugEnabled()) {
                LOG.debug("CANDIDATE: " + c);
                LOG.debug("CANDIDATE PARAMS: " + Arrays.toString(cTypes));
            }
            if (params.length != cTypes.length) continue;
            
            for (int i = 0; i < params.length; i++) {
                List<Class<?>> cSuper = ClassUtil.getSuperClasses(cTypes[i]);
                if (LOG.isDebugEnabled()) LOG.debug("  SUPER[" + cTypes[i].getSimpleName() + "] => " + cSuper);
                if (CollectionUtils.intersection(paramSuper[i], cSuper).isEmpty() == false) {
                    return ((Constructor<T>)c);
                }
            } // FOR (param)
        } // FOR (constructors)
        throw new RuntimeException("Failed to retrieve constructor for " + target_class.getSimpleName(), error);
    }
    
    /** Create an object for the given class and initialize it from conf
    *
    * @param theClass class of which an object is created
    * @param expected the expected parent class or interface
    * @return a new object
    */
   public static <T> T newInstance(Class<?> theClass, Class<T> expected) {
     T result;
     try {
       if (!expected.isAssignableFrom(theClass)) {
         throw new Exception("Specified class " + theClass.getName() + "" +
             "does not extend/implement " + expected.getName());
       }
       Class<? extends T> clazz = (Class<? extends T>)theClass;
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
     * 
     * @param class_name
     * @return
     */
    public static Class<?> getClass(String class_name) {
        Class<?> target_class = null;
        try {
            ClassLoader loader = ClassLoader.getSystemClassLoader();
            target_class = (Class<?>)loader.loadClass(class_name);
        } catch (Exception ex) {
            throw new RuntimeException("Failed to retrieve class for " + class_name, ex);
        }
        return (target_class);
 
    }
    
    /**
     * Returns true if asserts are enabled. This assumes that
     * we're always using the default system ClassLoader
     */
    public static boolean isAssertsEnabled() {
        boolean ret = false;
        try {
            assert(false);
        } catch (AssertionError ex) {
            ret = true;
        }
        return (ret);
}
}
