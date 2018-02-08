/*
 * OtterTune - JSONUtil.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.util;

import com.controller.util.json.JSONArray;
import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.controller.util.json.JSONStringer;
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

/** @author pavlo */
public abstract class JSONUtil {
  private static final Logger LOG = Logger.getLogger(JSONUtil.class.getName());

  private static final String JSON_CLASS_SUFFIX = "_class";
  private static final Map<Class<?>, Field[]> SERIALIZABLE_FIELDS =
      new HashMap<Class<?>, Field[]>();

  /**
   * @param clazz
   * @return
   */
  public static Field[] getSerializableFields(Class<?> clazz, String... fieldsToExclude) {
    Field[] ret = SERIALIZABLE_FIELDS.get(clazz);
    if (ret == null) {
      Collection<String> exclude = CollectionUtil.addAll(new HashSet<String>(), fieldsToExclude);
      synchronized (SERIALIZABLE_FIELDS) {
        ret = SERIALIZABLE_FIELDS.get(clazz);
        if (ret == null) {
          List<Field> fields = new ArrayList<Field>();
          for (Field f : clazz.getFields()) {
            int modifiers = f.getModifiers();
            if (Modifier.isTransient(modifiers) == false
                && Modifier.isPublic(modifiers) == true
                && Modifier.isStatic(modifiers) == false
                && exclude.contains(f.getName()) == false) {
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
   *
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
   *
   * @param <T>
   * @param object
   * @return
   */
  public static <T extends JSONSerializable> String format(T object) {
    JSONStringer stringer = new JSONStringer();
    try {
      if (object instanceof JSONObject) {
        return ((JSONObject) object).toString(2);
      }
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
   * @param <T>
   * @param object
   * @return
   */
  public static String toJSONString(Object object) {
    JSONStringer stringer = new JSONStringer();
    try {
      if (object instanceof JSONSerializable) {
        stringer.object();
        ((JSONSerializable) object).toJSON(stringer);
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
      JSONObject jsonObject = new JSONObject(json);
      t.fromJSON(jsonObject);
    } catch (JSONException ex) {
      throw new RuntimeException("Failed to deserialize object " + t, ex);
    }
    return (t);
  }

  /**
   * Write the contents of a JSONSerializable object out to a file on the local disk
   *
   * @param <T>
   * @param object
   * @param outputPath
   * @throws IOException
   */
  public static <T extends JSONSerializable> void save(T object, String outputPath)
      throws IOException {
    if (LOG.isDebugEnabled()) {
      LOG.debug(
          "Writing out contents of "
              + object.getClass().getSimpleName()
              + " to '"
              + outputPath
              + "'");
    }
    File f = new File(outputPath);
    try {
      FileUtil.makeDirIfNotExists(f.getParent());
      String json = object.toJSONString();
      FileUtil.writeStringToFile(f, format(json));
    } catch (Exception ex) {
      LOG.error(
          "Failed to serialize the " + object.getClass().getSimpleName() + " file '" + f + "'", ex);
      throw new IOException(ex);
    }
  }

  /**
   * Load in a JSONSerialable stored in a file
   *
   * @param <T>
   * @param object
   * @param inputPath
   * @throws Exception
   */
  public static <T extends JSONSerializable> void load(T object, String inputPath)
      throws IOException {
    if (LOG.isDebugEnabled()) {
      LOG.debug(
          "Loading in serialized "
              + object.getClass().getSimpleName()
              + " from '"
              + inputPath
              + "'");
    }
    String contents = FileUtil.readFile(inputPath);
    if (contents.isEmpty()) {
      throw new IOException(
          "The " + object.getClass().getSimpleName() + " file '" + inputPath + "' is empty");
    }
    try {
      object.fromJSON(new JSONObject(contents));
    } catch (Exception ex) {
      if (LOG.isDebugEnabled()) {
        LOG.error(
            "Failed to deserialize the "
                + object.getClass().getSimpleName()
                + " from file '"
                + inputPath
                + "'",
            ex);
      }
      throw new IOException(ex);
    }
    if (LOG.isDebugEnabled()) {
      LOG.debug("The loading of the " + object.getClass().getSimpleName() + " is complete");
    }
  }

  /**
   * For a given Enum, write out the contents of the corresponding field to the JSONObject We assume
   * that the given object has matching fields that correspond to the Enum members, except that
   * their names are lower case.
   *
   * @param <E>
   * @param <T>
   * @param stringer
   * @param object
   * @param baseClass
   * @param members
   * @throws JSONException
   */
  public static <E extends Enum<?>, T> void fieldsToJSON(
      JSONStringer stringer, T object, Class<? extends T> baseClass, E[] members)
      throws JSONException {
    try {
      fieldsToJSON(
          stringer, object, baseClass, ClassUtil.getFieldsFromMembersEnum(baseClass, members));
    } catch (NoSuchFieldException ex) {
      throw new JSONException(ex);
    }
  }

  /**
   * For a given list of Fields, write out the contents of the corresponding field to the JSONObject
   * The each of the JSONObject's elements will be the upper case version of the Field's name
   *
   * @param <T>
   * @param stringer
   * @param object
   * @param baseClass
   * @param fields
   * @throws JSONException
   */
  public static <T> void fieldsToJSON(
      JSONStringer stringer, T object, Class<? extends T> baseClass, Field[] fields)
      throws JSONException {
    if (LOG.isDebugEnabled()) {
      LOG.debug("Serializing out " + fields.length + " elements for " + baseClass.getSimpleName());
    }
    for (Field f : fields) {
      String jsonKey = f.getName().toUpperCase();
      stringer.key(jsonKey);

      try {
        Class<?> fclass = f.getType();
        Object fvalue = f.get(object);

        // Null
        if (fvalue == null) {
          writeFieldValue(stringer, fclass, fvalue);
          // Maps
        } else if (fvalue instanceof Map) {
          writeFieldValue(stringer, fclass, fvalue);
          // Everything else
        } else {
          writeFieldValue(stringer, fclass, fvalue);
        }
      } catch (Exception ex) {
        throw new JSONException(ex);
      }
    } // FOR
  }

  /**
   * @param stringer
   * @param fieldClass
   * @param fieldValue
   * @throws JSONException
   */
  public static void writeFieldValue(
      JSONStringer stringer, Class<?> fieldClass, Object fieldValue) throws JSONException {
    // Null
    if (fieldValue == null) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("writeNullFieldValue(" + fieldClass + ", " + fieldValue + ")");
      }
      stringer.value(null);

      // Collections
    } else if (ClassUtil.getInterfaces(fieldClass).contains(Collection.class)) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("writeCollectionFieldValue(" + fieldClass + ", " + fieldValue + ")");
      }
      stringer.array();
      for (Object value : (Collection<?>) fieldValue) {
        if (value == null) {
          stringer.value(null);
        } else {
          writeFieldValue(stringer, value.getClass(), value);
        }
      } // FOR
      stringer.endArray();

      // Maps
    } else if (fieldValue instanceof Map) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("writeMapFieldValue(" + fieldClass + ", " + fieldValue + ")");
      }
      stringer.object();
      for (Entry<?, ?> e : ((Map<?, ?>) fieldValue).entrySet()) {
        // We can handle null keys
        String keyValue = null;
        if (e.getKey() != null) {
          // deserialize it on the other side
          Class<?> keyClass = e.getKey().getClass();
          keyValue = makePrimitiveValue(keyClass, e.getKey()).toString();
        }
        stringer.key(keyValue);

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
      if (LOG.isDebugEnabled()) {
        LOG.debug("writePrimitiveFieldValue(" + fieldClass + ", " + fieldValue + ")");
      }
      stringer.value(makePrimitiveValue(fieldClass, fieldValue));
    }
    return;
  }

  /**
   * Read data from the given JSONObject and populate the given Map
   *
   * @param jsonObject
   * @param map
   * @param innerClasses
   * @throws Exception
   */
  @SuppressWarnings("unchecked")
  protected static void readMapField(
      final JSONObject jsonObject, final Map map, final Stack<Class> innerClasses)
      throws Exception {
    Class<?> keyClass = innerClasses.pop();
    Class<?> valClass = innerClasses.pop();
    Collection<Class<?>> valInterfaces = ClassUtil.getInterfaces(valClass);

    assert (jsonObject != null);
    for (String jsonKey : CollectionUtil.iterable(jsonObject.keys())) {
      final Stack<Class> nextInnerClasses = new Stack<Class>();
      nextInnerClasses.addAll(innerClasses);
      assert (nextInnerClasses.equals(innerClasses));

      // KEY
      Object key = JSONUtil.getPrimitiveValue(jsonKey, keyClass);

      // VALUE
      Object object = null;
      if (jsonObject.isNull(jsonKey)) {
        // Nothing...
      } else if (valInterfaces.contains(List.class)) {
        object = new ArrayList();
        readCollectionField(
            jsonObject.getJSONArray(jsonKey), (Collection) object, nextInnerClasses);
      } else if (valInterfaces.contains(Set.class)) {
        object = new HashSet();
        readCollectionField(
            jsonObject.getJSONArray(jsonKey), (Collection) object, nextInnerClasses);
      } else if (valInterfaces.contains(Map.class)) {
        object = new HashMap();
        readMapField(jsonObject.getJSONObject(jsonKey), (Map) object, nextInnerClasses);
      } else {
        String jsonString = jsonObject.getString(jsonKey);
        try {
          object = JSONUtil.getPrimitiveValue(jsonString, valClass);
        } catch (Exception ex) {
          System.err.println("val_interfaces: " + valInterfaces);
          LOG.error(
              "Failed to deserialize value '"
                  + jsonString
                  + "' from inner map key '"
                  + jsonKey
                  + "'");
          throw ex;
        }
      }
      map.put(key, object);
    }
  }

  /**
   * Read data from the given JSONArray and populate the given Collection
   *
   * @param jsonArray
   * @param collection
   * @param innerClasses
   * @throws Exception
   */
  @SuppressWarnings("unchecked")
  protected static void readCollectionField(
      final JSONArray jsonArray, final Collection collection, final Stack<Class> innerClasses)
      throws Exception {
    // We need to figure out what the inner type of the collection is
    // If it's a Collection or a Map, then we need to instantiate it before
    // we can call readFieldValue() again for it.
    Class innerClass = innerClasses.pop();
    Collection<Class<?>> innerInterfaces = ClassUtil.getInterfaces(innerClass);

    for (int i = 0, cnt = jsonArray.length(); i < cnt; i++) {
      final Stack<Class> nextInnerClasses = new Stack<Class>();
      nextInnerClasses.addAll(innerClasses);
      assert (nextInnerClasses.equals(innerClasses));
      Object value = null;

      // Null
      if (jsonArray.isNull(i)) {
        value = null;
        // Lists
      } else if (innerInterfaces.contains(List.class)) {
        value = new ArrayList();
        readCollectionField(jsonArray.getJSONArray(i), (Collection) value, nextInnerClasses);
        // Sets
      } else if (innerInterfaces.contains(Set.class)) {
        value = new HashSet();
        readCollectionField(jsonArray.getJSONArray(i), (Collection) value, nextInnerClasses);
        // Maps
      } else if (innerInterfaces.contains(Map.class)) {
        value = new HashMap();
        readMapField(jsonArray.getJSONObject(i), (Map) value, nextInnerClasses);
        // Values
      } else {
        String jsonString = jsonArray.getString(i);
        value = JSONUtil.getPrimitiveValue(jsonString, innerClass);
      }
      collection.add(value);
    } // FOR
    return;
  }

  /**
   * @param jsonObject
   * @param jsonKey
   * @param fieldHandle
   * @param object
   * @throws Exception
   */
  @SuppressWarnings("unchecked")
  public static void readFieldValue(
      final JSONObject jsonObject, final String jsonKey, Field fieldHandle, Object object)
      throws Exception {
    assert (jsonObject.has(jsonKey)) : "No entry exists for '" + jsonKey + "'";
    Class<?> fieldClass = fieldHandle.getType();
    Object fieldObject = fieldHandle.get(object);
    // String field_name = field_handle.getName();

    // Null
    if (jsonObject.isNull(jsonKey)) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Field " + jsonKey + " is null");
      }
      fieldHandle.set(object, null);

      // Collections
    } else if (ClassUtil.getInterfaces(fieldClass).contains(Collection.class)) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Field " + jsonKey + " is a collection");
      }
      assert (fieldObject != null);
      Stack<Class> innerClasses = new Stack<Class>();
      innerClasses.addAll(ClassUtil.getGenericTypes(fieldHandle));
      Collections.reverse(innerClasses);

      JSONArray jsonInner = jsonObject.getJSONArray(jsonKey);
      if (jsonInner == null) {
        throw new JSONException("No array exists for '" + jsonKey + "'");
      }
      readCollectionField(jsonInner, (Collection) fieldObject, innerClasses);

      // Maps
    } else if (fieldObject instanceof Map) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Field " + jsonKey + " is a map");
      }
      assert (fieldObject != null);
      Stack<Class> innerClasses = new Stack<Class>();
      innerClasses.addAll(ClassUtil.getGenericTypes(fieldHandle));
      Collections.reverse(innerClasses);

      JSONObject jsonInner = jsonObject.getJSONObject(jsonKey);
      if (jsonInner == null) {
        throw new JSONException("No object exists for '" + jsonKey + "'");
      }
      readMapField(jsonInner, (Map) fieldObject, innerClasses);

      // Everything else...
    } else {
      Class explicitFieldClass = JSONUtil.getClassForField(jsonObject, jsonKey);
      if (explicitFieldClass != null) {
        fieldClass = explicitFieldClass;
        if (LOG.isDebugEnabled()) {
          LOG.debug(
              "Found explict field class " + fieldClass.getSimpleName() + " for " + jsonKey);
        }
      }
      if (LOG.isDebugEnabled()) {
        LOG.debug("Field " + jsonKey + " is primitive type " + fieldClass.getSimpleName());
      }
      Object value = JSONUtil.getPrimitiveValue(jsonObject.getString(jsonKey), fieldClass);
      fieldHandle.set(object, value);
      if (LOG.isDebugEnabled()) {
        LOG.debug("Set field " + jsonKey + " to '" + value + "'");
      }
    }
  }

  /**
   * For the given enum, load in the values from the JSON object into the current object This will
   * throw errors if a field is missing
   *
   * @param <E>
   * @param jsonObject
   * @param members
   * @throws JSONException
   */
  public static <E extends Enum<?>, T> void fieldsFromJSON(
      JSONObject jsonObject, T object, Class<? extends T> baseClass, E... members)
      throws JSONException {
    JSONUtil.fieldsFromJSON(jsonObject, object, baseClass, false, members);
  }

  /**
   * For the given enum, load in the values from the JSON object into the current object If
   * ignore_missing is false, then JSONUtil will not throw an error if a field is missing
   *
   * @param <E>
   * @param <T>
   * @param jsonObject
   * @param object
   * @param baseClass
   * @param ignoreMissing
   * @param members
   * @throws JSONException
   */
  public static <E extends Enum<?>, T> void fieldsFromJSON(
      JSONObject jsonObject,
      T object,
      Class<? extends T> baseClass,
      boolean ignoreMissing,
      E... members)
      throws JSONException {
    try {
      fieldsFromJSON(
          jsonObject,
          object,
          baseClass,
          ignoreMissing,
          ClassUtil.getFieldsFromMembersEnum(baseClass, members));
    } catch (NoSuchFieldException ex) {
      throw new JSONException(ex);
    }
  }

  /**
   * For the given list of Fields, load in the values from the JSON object into the current object
   * If ignore_missing is false, then JSONUtil will not throw an error if a field is missing
   *
   * @param <E>
   * @param <T>
   * @param jsonObject
   * @param object
   * @param baseClass
   * @param ignoreMissing
   * @param fields
   * @throws JSONException
   */
  public static <E extends Enum<?>, T> void fieldsFromJSON(
      JSONObject jsonObject,
      T object,
      Class<? extends T> baseClass,
      boolean ignoreMissing,
      Field... fields)
      throws JSONException {
    for (Field fieldHandle : fields) {
      String jsonKey = fieldHandle.getName().toUpperCase();
      if (LOG.isDebugEnabled()) {
        LOG.debug("Retreiving value for field '" + jsonKey + "'");
      }

      if (!jsonObject.has(jsonKey)) {
        String msg =
            "JSONObject for "
                + baseClass.getSimpleName()
                + " does not have key '"
                + jsonKey
                + "': "
                + CollectionUtil.list(jsonObject.keys());
        if (ignoreMissing) {
          if (LOG.isDebugEnabled()) {
            LOG.warn(msg);
          }
          continue;
        } else {
          throw new JSONException(msg);
        }
      }

      try {
        readFieldValue(jsonObject, jsonKey, fieldHandle, object);
      } catch (Exception ex) {
        // System.err.println(field_class + ": " + ClassUtil.getSuperClasses(field_class));
        LOG.error(
            "Unable to deserialize field '" + jsonKey + "' from " + baseClass.getSimpleName(),
            ex);
        throw new JSONException(ex);
      }
    } // FOR
  }

  /**
   * Return the class of a field if it was stored in the JSONObject along with the value If there is
   * no class information, then this will return null
   *
   * @param jsonObject
   * @param jsonKey
   * @return
   * @throws JSONException
   */
  private static Class<?> getClassForField(JSONObject jsonObject, String jsonKey)
      throws JSONException {
    Class<?> fieldClass = null;
    // Check whether we also stored the class
    if (jsonObject.has(jsonKey + JSON_CLASS_SUFFIX)) {
      try {
        fieldClass = ClassUtil.getClass(jsonObject.getString(jsonKey + JSON_CLASS_SUFFIX));
      } catch (Exception ex) {
        LOG.error("Failed to include class for field '" + jsonKey + "'", ex);
        throw new JSONException(ex);
      }
    }
    return (fieldClass);
  }

  /**
   * Return the proper serialization string for the given value
   *
   * @param fieldClass
   * @param fieldValue
   * @return
   */
  private static Object makePrimitiveValue(Class<?> fieldClass, Object fieldValue) {
    Object value = null;

    // Class
    if (fieldClass.equals(Class.class)) {
      value = ((Class<?>) fieldValue).getName();
      // JSONSerializable
    } else if (ClassUtil.getInterfaces(fieldClass).contains(JSONSerializable.class)) {
      // Just return the value back. The JSON library will take care of it
      //            System.err.println(field_class + ": " + field_value);
      value = fieldValue;
      // Everything else
    } else {
      value = fieldValue; // .toString();
    }
    return (value);
  }

  /**
   * For the given JSON string, figure out what kind of object it is and return it
   *
   * @param jsonValue
   * @param fieldClass
   * @return
   * @throws Exception
   */
  public static Object getPrimitiveValue(String jsonValue, Class<?> fieldClass) throws Exception {
    Object value = null;

    // Class
    if (fieldClass.equals(Class.class)) {
      value = ClassUtil.getClass(jsonValue);
      if (value == null) {
        throw new JSONException("Failed to get class from '" + jsonValue + "'");
      }
      // Enum
    } else if (fieldClass.isEnum()) {
      for (Object o : fieldClass.getEnumConstants()) {
        Enum<?> e = (Enum<?>) o;
        if (jsonValue.equals(e.name())) {
          return (e);
        }
      } // FOR
      throw new JSONException(
          "Invalid enum value '"
              + jsonValue
              + "': "
              + Arrays.toString(fieldClass.getEnumConstants()));
      // JSONSerializable
    } else if (ClassUtil.getInterfaces(fieldClass).contains(JSONSerializable.class)) {
      value = ClassUtil.newInstance(fieldClass, null, null);
      ((JSONSerializable) value).fromJSON(new JSONObject(jsonValue));
      // Boolean
    } else if (fieldClass.equals(Boolean.class) || fieldClass.equals(boolean.class)) {
      // We have to use field_class.equals() because the value may be null
      value = Boolean.parseBoolean(jsonValue);
      // Short
    } else if (fieldClass.equals(Short.class) || fieldClass.equals(short.class)) {
      value = Short.parseShort(jsonValue);
      // Integer
    } else if (fieldClass.equals(Integer.class) || fieldClass.equals(int.class)) {
      value = Integer.parseInt(jsonValue);
      // Long
    } else if (fieldClass.equals(Long.class) || fieldClass.equals(long.class)) {
      value = Long.parseLong(jsonValue);
      // Float
    } else if (fieldClass.equals(Float.class) || fieldClass.equals(float.class)) {
      value = Float.parseFloat(jsonValue);
      // Double
    } else if (fieldClass.equals(Double.class) || fieldClass.equals(double.class)) {
      value = Double.parseDouble(jsonValue);
      // String
    } else if (fieldClass.equals(String.class)) {
      value = jsonValue.toString();
    }
    return (value);
  }

  public static Class<?> getPrimitiveType(String jsonValue) {
    Object value = null;

    // CHECKSTYLE:OFF
    // Class
    try {
      value = ClassUtil.getClass(jsonValue);
      if (value != null) return (Class.class);
    } catch (Throwable ex) {
    } // IGNORE

    // Short
    try {
      value = Short.parseShort(jsonValue);
      return (Short.class);
    } catch (NumberFormatException ex) {
    } // IGNORE

    // Integer
    try {
      value = Integer.parseInt(jsonValue);
      return (Integer.class);
    } catch (NumberFormatException ex) {
    } // IGNORE

    // Long
    try {
      value = Long.parseLong(jsonValue);
      return (Long.class);
    } catch (NumberFormatException ex) {
    } // IGNORE

    // Float
    try {
      value = Float.parseFloat(jsonValue);
      return (Float.class);
    } catch (NumberFormatException ex) {
    } // IGNORE

    // Double
    try {
      value = Double.parseDouble(jsonValue);
      return (Double.class);
    } catch (NumberFormatException ex) {
    } // IGNORE
    // CHECKSTYLE:ON

    // Boolean
    if (jsonValue.equalsIgnoreCase("true") || jsonValue.equalsIgnoreCase("false")) {
      return (Boolean.class);
    }

    // Default: String
    return (String.class);
  }
}
