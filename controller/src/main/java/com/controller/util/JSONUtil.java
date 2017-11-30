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

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;

import org.apache.log4j.Logger;

import com.controller.util.json.JSONArray;
import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.controller.util.json.JSONStringer;

/**
 * @author pavlo
 */
public abstract class JSONUtil {
    private static final Logger LOG = Logger.getLogger(JSONUtil.class.getName());
    
    private static final String JSON_CLASS_SUFFIX = "_class";
    private static final Map<Class<?>, Field[]> SERIALIZABLE_FIELDS = new HashMap<Class<?>, Field[]>();
    
    /**
     * 
     * @param clazz
     * @return
     */
    public static Field[] getSerializableFields(Class<?> clazz, String...fieldsToExclude) {
        Field ret[] = SERIALIZABLE_FIELDS.get(clazz);
        if (ret == null) {
            Collection<String> exclude = CollectionUtil.addAll(new HashSet<String>(), fieldsToExclude);
            synchronized (SERIALIZABLE_FIELDS) {
                ret = SERIALIZABLE_FIELDS.get(clazz);
                if (ret == null) {
                    List<Field> fields = new ArrayList<Field>();
                    for (Field f : clazz.getFields()) {
                        int modifiers = f.getModifiers();
                        if (Modifier.isTransient(modifiers) == false &&
                            Modifier.isPublic(modifiers) == true &&
                            Modifier.isStatic(modifiers) == false &&
                            exclude.contains(f.getName()) == false) {
                            fields.add(f);
                        }
                    } // FOR
                    ret = fields.toArray(new Field[0]);
                    SERIALIZABLE_FIELDS.put(clazz, ret);
                }
            } // SYNCH
        }
        return (ret);
    }
    
    /**
     * JSON Pretty Print
     * @param json
     * @return
     * @throws JSONException
     */
    public static String format(String json) {
        try {
            return (JSONUtil.format(new JSONObject(json)));
        } catch (RuntimeException ex) {
            throw ex;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
    
    /**
     * JSON Pretty Print
     * @param <T>
     * @param object
     * @return
     */
    public static <T extends JSONSerializable> String format(T object) {
        JSONStringer stringer = new JSONStringer();
        try {
            if (object instanceof JSONObject) return ((JSONObject)object).toString(2);
            stringer.object();
            object.toJSON(stringer);
            stringer.endObject();
        } catch (JSONException ex) {
            throw new RuntimeException(ex);
        }
        return (JSONUtil.format(stringer.toString()));
    }
    
    public static String format(JSONObject o) {
        try {
            return o.toString(1);
        } catch (JSONException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    /**
     * 
     * @param <T>
     * @param object
     * @return
     */
    public static String toJSONString(Object object) {
        JSONStringer stringer = new JSONStringer();
        try {
            if (object instanceof JSONSerializable) {
                stringer.object();
                ((JSONSerializable)object).toJSON(stringer);
                stringer.endObject();
            } else if (object != null) {
                Class<?> clazz = object.getClass();
//                stringer.key(clazz.getSimpleName());
                JSONUtil.writeFieldValue(stringer, clazz, object);
            }
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }
        return (stringer.toString());
    }
    
    public static <T extends JSONSerializable> T fromJSONString(T t, String json) {
        try {
            JSONObject json_object = new JSONObject(json);
            t.fromJSON(json_object);
        } catch (JSONException ex) {
            throw new RuntimeException("Failed to deserialize object " + t, ex);
        }
        return (t);
    }
    
    /**
     * Write the contents of a JSONSerializable object out to a file on the local disk
     * @param <T>
     * @param object
     * @param output_path
     * @throws IOException
     */
    public static <T extends JSONSerializable> void save(T object, String output_path) throws IOException {
        if (LOG.isDebugEnabled()) LOG.debug("Writing out contents of " + object.getClass().getSimpleName() + " to '" + output_path + "'");
        File f = new File(output_path);
        try {
            FileUtil.makeDirIfNotExists(f.getParent());
            String json = object.toJSONString();
            FileUtil.writeStringToFile(f, format(json));
        } catch (Exception ex) {
            LOG.error("Failed to serialize the " + object.getClass().getSimpleName() + " file '" + f + "'", ex);
            throw new IOException(ex);
        }
    }
    
    /**
     * Load in a JSONSerialable stored in a file
     * @param <T>
     * @param object
     * @param output_path
     * @throws Exception
     */
    public static <T extends JSONSerializable> void load(T object, String input_path) throws IOException {
        if (LOG.isDebugEnabled()) LOG.debug("Loading in serialized " + object.getClass().getSimpleName() + " from '" + input_path + "'");
        String contents = FileUtil.readFile(input_path);
        if (contents.isEmpty()) {
            throw new IOException("The " + object.getClass().getSimpleName() + " file '" + input_path + "' is empty");
        }
        try {
            object.fromJSON(new JSONObject(contents));
        } catch (Exception ex) {
            if (LOG.isDebugEnabled()) LOG.error("Failed to deserialize the " + object.getClass().getSimpleName() + " from file '" + input_path + "'", ex);
            throw new IOException(ex);
        }
        if (LOG.isDebugEnabled()) LOG.debug("The loading of the " + object.getClass().getSimpleName() + " is complete");
    }

    /**
     * For a given Enum, write out the contents of the corresponding field to the JSONObject
     * We assume that the given object has matching fields that correspond to the Enum members, except
     * that their names are lower case.
     * @param <E>
     * @param <T>
     * @param stringer
     * @param object
     * @param base_class
     * @param members
     * @throws JSONException
     */
    public static <E extends Enum<?>, T> void fieldsToJSON(JSONStringer stringer, T object, Class<? extends T> base_class, E members[]) throws JSONException {
        try {
            fieldsToJSON(stringer, object, base_class, ClassUtil.getFieldsFromMembersEnum(base_class, members));    
        } catch (NoSuchFieldException ex) {
            throw new JSONException(ex);
        }
    }
    
    /**
     * For a given list of Fields, write out the contents of the corresponding field to the JSONObject
     * The each of the JSONObject's elements will be the upper case version of the Field's name
     * @param <T>
     * @param stringer
     * @param object
     * @param base_class
     * @param fields
     * @throws JSONException
     */
    public static <T> void fieldsToJSON(JSONStringer stringer, T object, Class<? extends T> base_class, Field fields[]) throws JSONException {
        if (LOG.isDebugEnabled()) LOG.debug("Serializing out " + fields.length + " elements for " + base_class.getSimpleName());
        for (Field f : fields) {
            String json_key = f.getName().toUpperCase();
            stringer.key(json_key);
            
            try {
                Class<?> f_class = f.getType();
                Object f_value = f.get(object);
                
                // Null
                if (f_value == null) {
                    writeFieldValue(stringer, f_class, f_value);
                // Maps
                } else if (f_value instanceof Map) {
                    writeFieldValue(stringer, f_class, f_value);
                // Everything else
                } else {
                    writeFieldValue(stringer, f_class, f_value);
                }
            } catch (Exception ex) {
                throw new JSONException(ex);
            }
        } // FOR
    }
    
    /**
     * 
     * @param stringer
     * @param field_class
     * @param field_value
     * @throws JSONException
     */
    public static void writeFieldValue(JSONStringer stringer, Class<?> field_class, Object field_value) throws JSONException {
        // Null
        if (field_value == null) {
            if (LOG.isDebugEnabled()) LOG.debug("writeNullFieldValue(" + field_class + ", " + field_value + ")");
            stringer.value(null);
            
        // Collections
        } else if (ClassUtil.getInterfaces(field_class).contains(Collection.class)) {
            if (LOG.isDebugEnabled()) LOG.debug("writeCollectionFieldValue(" + field_class + ", " + field_value + ")");
            stringer.array();
            for (Object value : (Collection<?>)field_value) {
                if (value == null) {
                    stringer.value(null);
                } else {
                    writeFieldValue(stringer, value.getClass(), value);
                }
            } // FOR
            stringer.endArray();
            
        // Maps
        } else if (field_value instanceof Map) {
            if (LOG.isDebugEnabled()) LOG.debug("writeMapFieldValue(" + field_class + ", " + field_value + ")");
            stringer.object();
            for (Entry<?, ?> e : ((Map<?, ?>)field_value).entrySet()) {
                // We can handle null keys
                String key_value = null;
                if (e.getKey() != null) {
                    // deserialize it on the other side
                    Class<?> key_class = e.getKey().getClass();
                    key_value = makePrimitiveValue(key_class, e.getKey()).toString();
                }
                stringer.key(key_value);
                
                // We can also handle null values. Where is your god now???
                if (e.getValue() == null) {
                    stringer.value(null);
                } else {
                    writeFieldValue(stringer, e.getValue().getClass(), e.getValue());
                }
            } // FOR
            stringer.endObject();
            
        // Primitive
        } else {
            if (LOG.isDebugEnabled()) LOG.debug("writePrimitiveFieldValue(" + field_class + ", " + field_value + ")");
            stringer.value(makePrimitiveValue(field_class, field_value));
        }
        return;
    }
    
    /**
     * Read data from the given JSONObject and populate the given Map
     * @param json_object
     * @param catalog_db
     * @param map
     * @param inner_classes
     * @throws Exception
     */
    @SuppressWarnings("unchecked")
    protected static void readMapField(final JSONObject json_object, final Map map, final Stack<Class> inner_classes) throws Exception {
        Class<?> key_class = inner_classes.pop();
        Class<?> val_class = inner_classes.pop();
        Collection<Class<?>> val_interfaces = ClassUtil.getInterfaces(val_class);
        
        assert(json_object != null);
        for (String json_key : CollectionUtil.iterable(json_object.keys())) {
            final Stack<Class> next_inner_classes = new Stack<Class>();
            next_inner_classes.addAll(inner_classes);
            assert(next_inner_classes.equals(inner_classes));
            
            // KEY
            Object key = JSONUtil.getPrimitiveValue(json_key, key_class);
            
            // VALUE
            Object object = null;
            if (json_object.isNull(json_key)) {
                // Nothing...
            } else if (val_interfaces.contains(List.class)) {
                object = new ArrayList();
                readCollectionField(json_object.getJSONArray(json_key), (Collection)object, next_inner_classes);
            } else if (val_interfaces.contains(Set.class)) {
                object = new HashSet();
                readCollectionField(json_object.getJSONArray(json_key), (Collection)object, next_inner_classes);
            } else if (val_interfaces.contains(Map.class)) {
                object = new HashMap();
                readMapField(json_object.getJSONObject(json_key), (Map)object, next_inner_classes);
            } else {
                String json_string = json_object.getString(json_key);
                try {
                    object = JSONUtil.getPrimitiveValue(json_string, val_class);
                } catch (Exception ex) {
                    System.err.println("val_interfaces: " + val_interfaces);
                    LOG.error("Failed to deserialize value '" + json_string + "' from inner map key '" + json_key + "'");
                    throw ex;
                }
            }
            map.put(key, object);
        }
    }
    
    /**
     * Read data from the given JSONArray and populate the given Collection
     * @param json_array
     * @param catalog_db
     * @param collection
     * @param inner_classes
     * @throws Exception
     */
    @SuppressWarnings("unchecked")
    protected static void readCollectionField(final JSONArray json_array, final Collection collection, final Stack<Class> inner_classes) throws Exception {
        // We need to figure out what the inner type of the collection is
        // If it's a Collection or a Map, then we need to instantiate it before 
        // we can call readFieldValue() again for it.
        Class inner_class = inner_classes.pop();
        Collection<Class<?>> inner_interfaces = ClassUtil.getInterfaces(inner_class);
        
        for (int i = 0, cnt = json_array.length(); i < cnt; i++) {
            final Stack<Class> next_inner_classes = new Stack<Class>();
            next_inner_classes.addAll(inner_classes);
            assert(next_inner_classes.equals(inner_classes));
            Object value = null;
            
            // Null
            if (json_array.isNull(i)) {
                value = null;
            // Lists
            } else if (inner_interfaces.contains(List.class)) {
                value = new ArrayList();
                readCollectionField(json_array.getJSONArray(i), (Collection)value, next_inner_classes);
            // Sets
            } else if (inner_interfaces.contains(Set.class)) {
                value = new HashSet();
                readCollectionField(json_array.getJSONArray(i), (Collection)value, next_inner_classes);
            // Maps
            } else if (inner_interfaces.contains(Map.class)) {
                value = new HashMap();
                readMapField(json_array.getJSONObject(i), (Map)value, next_inner_classes);
            // Values
            } else {
                String json_string = json_array.getString(i);
                value = JSONUtil.getPrimitiveValue(json_string, inner_class);
            }
            collection.add(value);
        } // FOR
        return;
    }
    
    /**
     * 
     * @param json_object
     * @param catalog_db
     * @param json_key
     * @param field_handle
     * @param object
     * @throws Exception
     */
    @SuppressWarnings("unchecked")
    public static void readFieldValue(final JSONObject json_object, final String json_key, Field field_handle, Object object) throws Exception {
        assert(json_object.has(json_key)) : "No entry exists for '" + json_key + "'";
        Class<?> field_class = field_handle.getType();
        Object field_object = field_handle.get(object);
        // String field_name = field_handle.getName();
        
        // Null
        if (json_object.isNull(json_key)) {
            if (LOG.isDebugEnabled()) LOG.debug("Field " + json_key + " is null");
            field_handle.set(object, null);
            
        // Collections
        } else if (ClassUtil.getInterfaces(field_class).contains(Collection.class)) {
            if (LOG.isDebugEnabled()) LOG.debug("Field " + json_key + " is a collection");
            assert(field_object != null);
            Stack<Class> inner_classes = new Stack<Class>();
            inner_classes.addAll(ClassUtil.getGenericTypes(field_handle));
            Collections.reverse(inner_classes);
            
            JSONArray json_inner =json_object.getJSONArray(json_key); 
            if (json_inner == null) throw new JSONException("No array exists for '" + json_key + "'");
            readCollectionField(json_inner, (Collection)field_object, inner_classes);

        // Maps
        } else if (field_object instanceof Map) {
            if (LOG.isDebugEnabled()) LOG.debug("Field " + json_key + " is a map");
            assert(field_object != null);
            Stack<Class> inner_classes = new Stack<Class>();
            inner_classes.addAll(ClassUtil.getGenericTypes(field_handle));
            Collections.reverse(inner_classes);
            
            JSONObject json_inner = json_object.getJSONObject(json_key);
            if (json_inner == null) throw new JSONException("No object exists for '" + json_key + "'");
            readMapField(json_inner, (Map)field_object, inner_classes);
            
        // Everything else...
        } else {
            Class explicit_field_class = JSONUtil.getClassForField(json_object, json_key);
            if (explicit_field_class != null) {
                field_class = explicit_field_class;
                if (LOG.isDebugEnabled()) LOG.debug("Found explict field class " + field_class.getSimpleName() + " for " + json_key);
            }
            if (LOG.isDebugEnabled()) LOG.debug("Field " + json_key + " is primitive type " + field_class.getSimpleName());
            Object value = JSONUtil.getPrimitiveValue(json_object.getString(json_key), field_class);
            field_handle.set(object, value);
            if (LOG.isDebugEnabled()) LOG.debug("Set field " + json_key + " to '" + value + "'");
        }
    }
    
    /**
     * For the given enum, load in the values from the JSON object into the current object
     * This will throw errors if a field is missing
     * @param <E>
     * @param json_object
     * @param catalog_db
     * @param members
     * @throws JSONException
     */
    public static <E extends Enum<?>, T> void fieldsFromJSON(JSONObject json_object, T object, Class<? extends T> base_class, E...members) throws JSONException {
        JSONUtil.fieldsFromJSON(json_object, object, base_class, false, members);
    }
    
    /**
     * For the given enum, load in the values from the JSON object into the current object
     * If ignore_missing is false, then JSONUtil will not throw an error if a field is missing
     * @param <E>
     * @param <T>
     * @param json_object
     * @param catalog_db
     * @param object
     * @param base_class
     * @param ignore_missing
     * @param members
     * @throws JSONException
     */
    public static <E extends Enum<?>, T> void fieldsFromJSON(JSONObject json_object, T object, Class<? extends T> base_class, boolean ignore_missing, E...members) throws JSONException {
        try {
            fieldsFromJSON(json_object, object, base_class, ignore_missing, ClassUtil.getFieldsFromMembersEnum(base_class, members));    
        } catch (NoSuchFieldException ex) {
            throw new JSONException(ex);
        }
    }
    
    /**
     * For the given list of Fields, load in the values from the JSON object into the current object
     * If ignore_missing is false, then JSONUtil will not throw an error if a field is missing
     * @param <E>
     * @param <T>
     * @param json_object
     * @param catalog_db
     * @param object
     * @param base_class
     * @param ignore_missing
     * @param fields
     * @throws JSONException
     */
    public static <E extends Enum<?>, T> void fieldsFromJSON(JSONObject json_object, T object, Class<? extends T> base_class, boolean ignore_missing, Field...fields) throws JSONException {
        for (Field field_handle : fields) {
            String json_key = field_handle.getName().toUpperCase();
            if (LOG.isDebugEnabled()) LOG.debug("Retreiving value for field '" + json_key + "'");
            
            if (!json_object.has(json_key)) {
                String msg = "JSONObject for " + base_class.getSimpleName() + " does not have key '" + json_key + "': " + CollectionUtil.list(json_object.keys()); 
                if (ignore_missing) {
                    if (LOG.isDebugEnabled()) LOG.warn(msg);
                    continue;
                } else {
                    throw new JSONException(msg);    
                }
            }
            
            try {
                readFieldValue(json_object, json_key, field_handle, object);
            } catch (Exception ex) {
                // System.err.println(field_class + ": " + ClassUtil.getSuperClasses(field_class));
                LOG.error("Unable to deserialize field '" + json_key + "' from " + base_class.getSimpleName(), ex);
                throw new JSONException(ex);
            }
        } // FOR
    }

    /**
     * Return the class of a field if it was stored in the JSONObject along with the value
     * If there is no class information, then this will return null 
     * @param json_object
     * @param json_key
     * @return
     * @throws JSONException
     */
    private static Class<?> getClassForField(JSONObject json_object, String json_key) throws JSONException {
        Class<?> field_class = null;
        // Check whether we also stored the class
        if (json_object.has(json_key + JSON_CLASS_SUFFIX)) {
            try {
                field_class = ClassUtil.getClass(json_object.getString(json_key + JSON_CLASS_SUFFIX));
            } catch (Exception ex) {
                LOG.error("Failed to include class for field '" + json_key + "'", ex);
                throw new JSONException(ex);
            }
        }
        return (field_class);
        
    }
    
    /**
     * Return the proper serialization string for the given value
     * @param field_name
     * @param field_class
     * @param field_value
     * @return
     */
    private static Object makePrimitiveValue(Class<?> field_class, Object field_value) {
        Object value = null;
        
        // Class
        if (field_class.equals(Class.class)) {
            value = ((Class<?>)field_value).getName();
        // JSONSerializable
        } else if (ClassUtil.getInterfaces(field_class).contains(JSONSerializable.class)) {
            // Just return the value back. The JSON library will take care of it
//            System.err.println(field_class + ": " + field_value);
            value = field_value;
        // Everything else
        } else {
            value = field_value; // .toString();
        }
        return (value);
    }

    
    /**
     * For the given JSON string, figure out what kind of object it is and return it
     * @param json_value
     * @param field_class
     * @param catalog_db
     * @return
     * @throws Exception
     */
    public static Object getPrimitiveValue(String json_value, Class<?> field_class) throws Exception {
        Object value = null;

        // Class
        if (field_class.equals(Class.class)) {
            value = ClassUtil.getClass(json_value);
            if (value == null) throw new JSONException("Failed to get class from '" + json_value + "'");
        // Enum
        } else if (field_class.isEnum()) {
            for (Object o : field_class.getEnumConstants()) {
                Enum<?> e = (Enum<?>)o;
                if (json_value.equals(e.name())) return (e);
            } // FOR
            throw new JSONException("Invalid enum value '" + json_value + "': " + Arrays.toString(field_class.getEnumConstants()));
         // JSONSerializable
        } else if (ClassUtil.getInterfaces(field_class).contains(JSONSerializable.class)) {
            value = ClassUtil.newInstance(field_class, null, null);
            ((JSONSerializable)value).fromJSON(new JSONObject(json_value));
        // Boolean
        } else if (field_class.equals(Boolean.class) || field_class.equals(boolean.class)) {
            // We have to use field_class.equals() because the value may be null
            value = Boolean.parseBoolean(json_value);
        // Short
        } else if (field_class.equals(Short.class) || field_class.equals(short.class)) {
            value = Short.parseShort(json_value);
        // Integer
        } else if (field_class.equals(Integer.class) || field_class.equals(int.class)) {
            value = Integer.parseInt(json_value);
        // Long
        } else if (field_class.equals(Long.class) || field_class.equals(long.class)) {
            value = Long.parseLong(json_value);
        // Float
        } else if (field_class.equals(Float.class) || field_class.equals(float.class)) {
            value = Float.parseFloat(json_value);
        // Double
        } else if (field_class.equals(Double.class) || field_class.equals(double.class)) {
            value = Double.parseDouble(json_value);
        // String
        } else if (field_class.equals(String.class)) {
            value = json_value.toString();
        }
        return (value);
    }
    
    public static Class<?> getPrimitiveType(String json_value) {
        Object value = null;
        
        // Class
        try {
            value = ClassUtil.getClass(json_value);
            if (value != null) return (Class.class);
        } catch (Throwable ex) { } // IGNORE
        
        // Short
        try {
            value = Short.parseShort(json_value);
            return (Short.class);
        } catch (NumberFormatException ex) { } // IGNORE
        
        // Integer
        try {
            value = Integer.parseInt(json_value);
            return (Integer.class);
        } catch (NumberFormatException ex) { } // IGNORE
        
        // Long
        try {
            value = Long.parseLong(json_value);
            return (Long.class);
        } catch (NumberFormatException ex) { } // IGNORE
        
        // Float
        try {
            value = Float.parseFloat(json_value);
            return (Float.class);
        } catch (NumberFormatException ex) { } // IGNORE
        
        // Double
        try {
            value = Double.parseDouble(json_value);
            return (Double.class);
        } catch (NumberFormatException ex) { } // IGNORE

        
        // Boolean
        if (json_value.equalsIgnoreCase("true") || json_value.equalsIgnoreCase("false"))
            return (Boolean.class);

        // Default: String
        return (String.class);
    }
}
