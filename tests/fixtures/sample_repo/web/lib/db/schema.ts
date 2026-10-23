import { pgTable, text, integer } from "drizzle-orm/pg-core";

export const user = pgTable("user", {
  id: text("id").primaryKey(),
  email: text("email").notNull(),
});

export const order = pgTable("order", {
  id: text("id").primaryKey(),
  userId: text("user_id").notNull(),
  total: integer("total").notNull(),
});

export type User = typeof user.$inferSelect;
