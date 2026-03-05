#!/bin/sh
set -e
echo "Running database migrations..."
npx prisma migrate deploy
echo "Running database seed..."
node prisma/seed-deploy.js
echo "Starting server..."
exec node server.js
