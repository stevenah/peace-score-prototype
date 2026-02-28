#!/bin/sh
set -e
echo "Running database migrations..."
npx prisma db push
echo "Starting server..."
exec npm start
