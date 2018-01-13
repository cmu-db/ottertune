/******************************************************************************
 *  Copyright 2015-2018 by OLTPBenchmark Project                                   *
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

import org.apache.commons.collections15.set.ListOrderedSet;
import org.apache.commons.lang.NotImplementedException;

import java.util.*;
import java.util.Map.Entry;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;
import java.util.stream.Collectors;

/**
 * 
 * @author pavlo
 */
public abstract class CollectionUtil {

    private static final Logger LOG = LogManager.getLogger(CollectionUtil.class);
    
    private static final Random RANDOM = new Random();
    
    /**
     * Put all of the elements in items into the given array
     * This assumes that the array has been pre-allocated
     * @param <T>
     * @param items
     * @param array
     */
    public static <T> void toArray(Collection<T> items, Object array[], boolean convert_to_primitive) {
        assert(items.size() == array.length);
        
        int i = 0;
        for (T t : items) {
            if (convert_to_primitive) {
                if (t instanceof Long) {
                    array[i] = ((Long)t).longValue();
                } else if (t instanceof Integer) {
                    array[i] = ((Integer)t).intValue();
                } else if (t instanceof Double) {
                    array[i] = ((Double)t).doubleValue();
                } else if (t instanceof Boolean) {
                    array[i] = ((Boolean)t).booleanValue();
                }
            } else {
                array[i] = t;
            }
        }
    }
    
    public static int[] toIntArray(Collection<Integer> items) {
        return items.stream().mapToInt(x -> x).toArray();
    }
    
    /**
     * Put all the values of an Iterator into a List
     * @param <T>
     * @param it
     * @return
     */
    public static <T> List<T> list(Iterator<T> it) {
        List<T> list = new ArrayList<>();
        CollectionUtil.addAll(list, it);
        return list;
    }
    /**
     * Put all of the values of an Enumeration into a new List
     * @param <T>
     * @param e
     * @return
     */
    public static <T> List<T> list(Enumeration<T> e) {
        return list(iterable(e));
    }
    /**
     * Put all of the values of an Iterable into a new List
     * @param <T>
     * @param it
     * @return
     */
    public static <T> List<T> list(Iterable<T> it) {
        return (list(it.iterator()));
    }
    
    /**
     * Put all the values of an Iterator into a Set
     * @param <T>
     * @param it
     * @return
     */
    public static <T> Set<T> set(Iterator<T> it) {
        Set<T> set = new HashSet<T>();
        CollectionUtil.addAll(set, it);
        return set;
    }
    /**
     * Put all of the values of an Iterable into a new Set
     * @param <T>
     * @param it
     * @return
     */
    public static <T> Set<T> set(Iterable<T> it) {
        return set(it.iterator());
    }
    
    /**
     * Returns a list containing the string representations of the elements in the collection
     * @param <T>
     * @param data
     * @return
     */
    public static <T> List<String> toStringList(Collection<T> data) {
        return data.stream().map(Object::toString).collect(Collectors.toList());
    }
    
    /**
     * Returns a set containing the string representations of the elements in the collection 
     * @param <T>
     * @param data
     * @return
     */
    public static <T> Set<String> toStringSet(Collection<T> data) {
        return data.stream().map(Object::toString).collect(Collectors.toSet());
    }
    
    /**
     * Return a random value from the given Collection
     * @param <T>
     * @param items
     */
    public static <T> T random(Collection<T> items) {
        return CollectionUtil.random(items, RANDOM);
    }
    
    /**
     * Return a random value from the given Collection
     * @param <T>
     * @param items
     * @param rand
     * @return
     */
    public static <T> T random(Collection<T> items, Random rand) {
        int idx = rand.nextInt(items.size());
        return CollectionUtil.get(items, idx);
    }
    
    /**
     * Return a random value from the given Iterable
     * @param <T>
     * @param it
     * @return
     */
    public static <T> T random(Iterable<T> it) {
        return (CollectionUtil.random(it, RANDOM));
    }
    
    /**
     * Return a random value from the given Iterable
     * @param <T>
     * @param it
     * @param rand
     * @return
     */
    public static <T> T random(Iterable<T> it, Random rand) {
        List<T> list = new ArrayList<>();
        it.forEach(list::add);
        return CollectionUtil.random(list, rand);
    }
    
    public static <E extends Enum<?>> Set<E> getAllExcluding(E elements[], E...excluding) {
        HashSet<E> excludingSet = new HashSet<>(Arrays.asList(excluding));
        HashSet<E> elementSet = new HashSet<>(Arrays.asList(elements));

        elementSet.removeAll(excludingSet);
        return elementSet;
    }
    
    /**
     * Add all the items in the array to a Collection
     * @param <T>
     * @param data
     * @param items
     */
    public static <T> Collection<T> addAll(Collection<T> data, T...items) {
        data.addAll(Arrays.asList(items));
        return data;
    }
    
    /**
     * Add all the items in the Enumeration into a Collection
     * @param <T>
     * @param data
     * @param items
     */
    public static <T> Collection<T> addAll(Collection<T> data, Enumeration<T> items) {
        iterable(items).forEach(data::add);
        return data;
    }
    
    /**
     * Add all of the items from the Iterable into the given collection
     * @param <T>
     * @param data
     * @param items
     */
    public static <T> Collection<T> addAll(Collection<T> data, Iterable<T> items) {
        return CollectionUtil.addAll(data, items.iterator());
    }
    
    /**
     * Add all of the items from the Iterator into the given collection
     * @param <T>
     * @param data
     * @param items
     */
    public static <T> Collection<T> addAll(Collection<T> data, Iterator<T> items) {
        items.forEachRemaining(data::add);
        return data;
    }
    
    /**
     * 
     * @param <T>
     * @param <U>
     * @param map
     * @return
     */
    public static <T, U extends Comparable<U>> T getGreatest(Map<T, U> map) {
        T max_key = null;
        U max_value = null;
        for (Entry<T, U> e:map.entrySet()) {
            T key = e.getKey();
            U value = e.getValue();
            if (max_value == null || value.compareTo(max_value) > 0) {
                max_value = value;
                max_key = key;
            }
        } // FOR
        return (max_key);
    }
    
    /**
     * Return the first item in a Iterable
     * @param <T>
     * @param items
     * @return
     */
    public static <T> T first(Iterable<T> items) {
        return CollectionUtil.get(items, 0);
    }
    
    /**
     * Return the first item in a Iterator
     * @param <T>
     * @param items
     * @return
     */
    public static <T> T first(Iterator<T> items) {
        return items.hasNext() ? items.next() : null;
    }
    
    /**
     * Returns the first item in an Enumeration
     * @param <T>
     * @param items
     * @return
     */
    public static <T> T first(Enumeration<T> items) {
        return items.hasMoreElements() ? items.nextElement() : null;
    }
    
    /**
     * Return the ith element of a set. Super lame
     * @param <T>
     * @param items
     * @param idx
     * @return
     */
    public static <T> T get(Iterable<T> items, int idx) {
        if (items instanceof AbstractList<?>) {
            return ((AbstractList<T>)items).get(idx);
        } else if (items instanceof ListOrderedSet<?>) {
            return ((ListOrderedSet<T>)items).get(idx);
        }
        int ctr = 0;
        for (T t : items) {
            if (ctr++ == idx) return (t);
        }
        return (null);
    }
    
    /**
     * Return the last item in an Iterable
     * @param <T>
     * @param items
     * @return
     */
    public static <T> T last(Iterable<T> items) {

        T last = null;
//        items.iterator().forEachRemaining(i -> last = i);

        if (items instanceof AbstractList<?>) {
            AbstractList<T> list = (AbstractList<T>)items;
            last = (list.isEmpty() ? null : list.get(list.size() - 1));
        } else {
            for (T t : items) {
                last = t;
            }
        }
        return (last);
    }
    
    /**
     * Return the last item in an array
     * @param <T>
     * @param items
     * @return
     */
    public static <T> T last(T...items) {
        return Arrays.stream(items).reduce((first, second) -> second).orElse(null);
    }

    /**
     * 
     * @param <K>
     * @param <V>
     * @param map
     * @return
     */
    public static <K extends Comparable<?>, V> List<V> getSortedList(SortedMap<K, Collection<V>> map) {
        return map.values().stream().flatMap(Collection::stream).collect(Collectors.toList());
    }
    
    /**
     * Wrap an Iterable around an Iterator
     * @param <T>
     * @param it
     * @return
     */
    public static <T> Iterable<T> iterable(final Iterator<T> it) {
        return new Iterable<T>() {
            @Override
            public Iterator<T> iterator() {
                return it;
            }
        };
    }
    
    public static <T> Iterable<T> iterable(final T values[]) {
        return (new Iterable<T>() {
            @Override
            public Iterator<T> iterator() {
                return new Iterator<T>() {
                    private int idx = 0;
                    @Override
                    public boolean hasNext() {
                        return (this.idx < values.length);
                    }
                    @Override
                    public T next() {
                        if (this.idx == values.length)
                            throw new NoSuchElementException();
                        return values[this.idx++];
                    }
                    @Override
                    public void remove() {
                        throw new UnsupportedOperationException();
                    }
                };
            }
        });
    }
    
    /**
     * Wrap an Iterable around an Enumeration
     * @param <T>
     * @param e
     * @return
     */
    public static <T> Iterable<T> iterable(final Enumeration<T> e) {
        return (new Iterable<T>() {
            @Override
            public Iterator<T> iterator() {
                return new Iterator<T>() {
                    @Override
                    public boolean hasNext() {
                        return (e.hasMoreElements());
                    }
                    @Override
                    public T next() {
                        return (e.nextElement());
                    }
                    @Override
                    public void remove() {
                        throw new NotImplementedException();
                    }
                };
            }
        });
    }
    
    public static <T> T pop(Collection<T> items) {
        T t = CollectionUtil.first(items);
        if (t != null) {
            boolean ret = items.remove(t);
            assert(ret);
        }
        return (t);
    }
}
