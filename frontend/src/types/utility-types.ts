/**
 * Utility types for TypeScript
 */

/**
 * Makes specified properties in T required
 */
export type RequireProps<T, K extends keyof T> = T & { 
  [P in K]-?: T[P] 
};

/**
 * Makes specified properties in T optional
 */
export type OptionalProps<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Creates a new type from T with only the specified keys
 */
export type PickProps<T, K extends keyof T> = Pick<T, K>;

/**
 * Creates a type that requires at least one of the properties from T
 */
export type RequireAtLeastOne<T, Keys extends keyof T = keyof T> =
  Pick<T, Exclude<keyof T, Keys>> 
  & {
      [K in Keys]-?: Required<Pick<T, K>> & Partial<Pick<T, Exclude<Keys, K>>>
    }[Keys];

/**
 * Creates a type that requires exactly one of the properties from T
 */
export type RequireExactlyOne<T, Keys extends keyof T = keyof T> =
  Pick<T, Exclude<keyof T, Keys>> 
  & {
      [K in Keys]: Required<Pick<T, K>> & Partial<Record<Exclude<Keys, K>, undefined>>
    }[Keys];

/**
 * Makes all properties of T mutable (removes readonly)
 */
export type Mutable<T> = {
  -readonly [P in keyof T]: T[P]
};

/**
 * Makes all properties of T readonly
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: DeepReadonly<T[P]>
};

/**
 * Creates a nullable version of type T
 */
export type Nullable<T> = T | null;

/**
 * Creates a type that represents T or is undefined
 */
export type Optional<T> = T | undefined;

/**
 * Creates a deep partial type of T (all nested properties are optional)
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Creates a record from a union type
 */
export type UnionToRecord<T extends string | number | symbol, V> = {
  [K in T]: V;
};

/**
 * Gets keys of T where the value type is assignable to U
 */
export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

/**
 * Creates a discriminated union based on a property
 */
export type DiscriminateUnion<T, K extends keyof T, V extends T[K]> = 
  T extends { [key in K]: V } ? T : never;

/**
 * Allows for different return types based on input
 */
export type FunctionReturnType<T> = 
  T extends (...args: any[]) => infer R ? R : never; 